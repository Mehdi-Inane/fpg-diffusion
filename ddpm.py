import argparse
import os

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import matplotlib.pyplot as plt
import numpy as np

import datasets
from positional_embeddings import PositionalEmbedding


class Block(nn.Module):
    def __init__(self, size: int,device='cuda'):
        super().__init__()
        self.device = device
        self.ff = nn.Linear(size, size).to(self.device)
        self.act = nn.GELU().to(self.device)

    def forward(self, x: torch.Tensor):
        return x + self.act(self.ff(x))


class MLP(nn.Module):
    def __init__(self, hidden_size: int = 128, hidden_layers: int = 3, emb_size: int = 128,
                 time_emb: str = "sinusoidal", input_emb: str = "sinusoidal",device='cuda'):
        super().__init__()
        self.device = device
        self.time_mlp = PositionalEmbedding(emb_size, time_emb,device=self.device)
        self.input_mlp1 = PositionalEmbedding(emb_size, input_emb, scale=25.0,device=self.device)
        self.input_mlp2 = PositionalEmbedding(emb_size, input_emb, scale=25.0,device=self.device)

        concat_size = len(self.time_mlp.layer) + \
            len(self.input_mlp1.layer) + len(self.input_mlp2.layer)
        layers = [nn.Linear(concat_size, hidden_size).to(self.device), nn.GELU().to(self.device)]
        for _ in range(hidden_layers):
            layers.append(Block(hidden_size,device=self.device))
        layers.append(nn.Linear(hidden_size, 2).to(self.device))
        self.joint_mlp = nn.Sequential(*layers)

    def forward(self, x, t):
        x1_emb = self.input_mlp1(x[:, 0])
        x2_emb = self.input_mlp2(x[:, 1])
        t_emb = self.time_mlp(t)
        x = torch.cat((x1_emb, x2_emb, t_emb), dim=-1)
        x = self.joint_mlp(x)
        return x
    

    def sample(self, num_samples, noise_scheduler, device='cuda',deterministic=False):
        self.eval()
        frames = []
        sample = torch.randn(num_samples, 2).to(device)
        timesteps = list(range(len(noise_scheduler)))[::-1]
        for i, t in enumerate(tqdm(timesteps)):
            t = torch.from_numpy(np.repeat(t, num_samples)).long()
            with torch.no_grad():
                residual = self.forward(sample, t.to(device))
            sample = noise_scheduler.step(residual, t[0], sample,deterministic=deterministic)
        frames.append(sample)
        return frames




    def restart(self,t_max,t_min,sample,noise_scheduler,K,device='cuda'):
        timesteps = list(range(len(noise_scheduler)))[::-1]
        print(f'timesteps {timesteps}')
        t_min_idx = timesteps.index(t_min)
        t_max_idx = timesteps.index(t_max)
        timesteps_restart = timesteps[t_min_idx:t_max_idx+1]
        print(f'restart ',timesteps_restart)
        for i in range(K):
            for j, t in enumerate(timesteps_restart):
                t = torch.from_numpy(np.repeat(t, sample.shape[0])).long()
                with torch.no_grad():
                    residual = self.forward(sample, t.to(device))
                sample = noise_scheduler.step(residual, t[0], sample,deterministic=True)
            noise = torch.randn(sample.shape).to(device)
            times =torch.tensor(t_max).repeat(sample.shape[0]).long()
            sample = noise_scheduler.add_noise(sample, noise, times)
        return sample
    




    def restart_sampling(self, num_samples, noise_scheduler, device='cuda',deterministic=True,K=4):
        self.eval()
        frames = []
        sample = torch.randn(num_samples, 2).to(device)
        timesteps = list(range(len(noise_scheduler)))[::-1]
        t_min,t_max = timesteps[1],timesteps[5]
        for i, t in enumerate(timesteps):
            if t == t_max:
                sample = self.restart(t_max,t_min,sample,noise_scheduler,K)
            t = torch.from_numpy(np.repeat(t, num_samples)).long()
            with torch.no_grad():
                residual = self.forward(sample, t.to(device))
            sample = noise_scheduler.step(residual, t[0], sample,deterministic=deterministic)
        frames.append(sample)
        return frames


    def sample_with_rl(self,num_samples,noise_scheduler,ppo_net,device='cuda'):
        self.eval()
        frames = []
        samples = torch.randn(num_samples, 2).to(device)
        timesteps = list(range(len(noise_scheduler)))[::-1]
        for i, _t in enumerate(tqdm(timesteps)):
            samples = samples.to(device)
            t = torch.from_numpy(np.repeat(_t, num_samples)).long().to(device)
            with torch.no_grad():
                residual = self(samples, t.to(device))
            action = ppo_net.get_action(samples,t.to(device))
            ode_indices = (action== 0).squeeze()
            if ode_indices.any(): # perform one step of ODE
                samples[ode_indices] =noise_scheduler.step(residual[ode_indices], t[0], samples[ode_indices],deterministic=False)
            noise_indices = (action == 1).squeeze()
            if noise_indices.any():
                noise = torch.randn_like(samples[noise_indices]).to(device)
                times = torch.tensor(1,device=device,dtype=torch.long).repeat(samples[noise_indices].shape[0])
                samples[noise_indices] = noise_scheduler.add_noise(samples[noise_indices],noise,times)
            if i == len(timesteps)-1:
                samples =noise_scheduler.step(residual, t[0], samples,deterministic=True) 
                break
        frames.append(samples)
        return frames


class NoiseScheduler():
    def __init__(self,
                 num_timesteps=1000,
                 beta_start=0.0001,
                 beta_end=0.02,
                 beta_schedule="linear",
                 device='cuda'):
        self.device = device
        self.num_timesteps = num_timesteps
        if beta_schedule == "linear":
            self.betas = torch.linspace(
                beta_start, beta_end, num_timesteps, dtype=torch.float32,device=self.device)
        elif beta_schedule == "quadratic":
            self.betas = torch.linspace(
                beta_start ** 0.5, beta_end ** 0.5, num_timesteps, dtype=torch.float32,device=self.device) ** 2

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0).to(self.device)
        self.alphas_cumprod_prev = F.pad(
            self.alphas_cumprod[:-1], (1, 0), value=1.).to(self.device)

        # required for self.add_noise
        self.sqrt_alphas_cumprod = self.alphas_cumprod ** 0.5
        self.sqrt_one_minus_alphas_cumprod = (1 - self.alphas_cumprod) ** 0.5

        # required for reconstruct_x0
        self.sqrt_inv_alphas_cumprod = torch.sqrt(1 / self.alphas_cumprod)
        self.sqrt_inv_alphas_cumprod_minus_one = torch.sqrt(
            1 / self.alphas_cumprod - 1)

        # required for q_posterior
        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1. - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1. - self.alphas_cumprod)

    def reconstruct_x0(self, x_t, t, noise):
        s1 = self.sqrt_inv_alphas_cumprod[t]
        s2 = self.sqrt_inv_alphas_cumprod_minus_one[t]
        s1 = s1.reshape(-1, 1)
        s2 = s2.reshape(-1, 1)
        return s1 * x_t - s2 * noise

    def q_posterior(self, x_0, x_t, t):
        s1 = self.posterior_mean_coef1[t]
        s2 = self.posterior_mean_coef2[t]
        s1 = s1.reshape(-1, 1)
        s2 = s2.reshape(-1, 1)
        mu = s1 * x_0 + s2 * x_t
        return mu

    def get_variance(self, t):
        if t == 0:
            return 0

        variance = self.betas[t] * (1. - self.alphas_cumprod_prev[t]) / (1. - self.alphas_cumprod[t])
        variance = variance.clip(1e-20)
        return variance

    def step(self, model_output, timestep, sample,deterministic=False):
        t = timestep
        pred_original_sample = self.reconstruct_x0(sample, t, model_output)
        pred_prev_sample = self.q_posterior(pred_original_sample, sample, t)

        variance = 0
        if t > 0:
            noise = torch.randn_like(model_output)
            variance = (self.get_variance(t) ** 0.5) * noise
            if deterministic:
                variance = 0

        pred_prev_sample = pred_prev_sample + variance

        return pred_prev_sample

    def add_noise(self, x_start, x_noise, timesteps):
        s1 = self.sqrt_alphas_cumprod[timesteps]
        s2 = self.sqrt_one_minus_alphas_cumprod[timesteps]

        s1 = s1.reshape(-1, 1)
        s2 = s2.reshape(-1, 1)
        return s1 * x_start + s2 * x_noise

    def __len__(self):
        return self.num_timesteps


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type=str, default="base")
    parser.add_argument("--dataset", type=str, default="gaussian_mixture", choices=["circle", "dino", "line", "moons","gaussian_mixture"])
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=1000)
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--num_timesteps", type=int, default=50)
    parser.add_argument("--beta_schedule", type=str, default="linear", choices=["linear", "quadratic"])
    parser.add_argument("--embedding_size", type=int, default=128)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--hidden_layers", type=int, default=3)
    parser.add_argument("--time_embedding", type=str, default="sinusoidal", choices=["sinusoidal", "learnable", "linear", "zero"])
    parser.add_argument("--input_embedding", type=str, default="sinusoidal", choices=["sinusoidal", "learnable", "linear", "identity"])
    parser.add_argument("--save_images_step", type=int, default=1)
    config = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if config.dataset != 'gaussian_mixture':
        dataset = datasets.get_dataset(config.dataset)
    else:
        dataset = datasets.DataSet(remove=True,total_len=10000)
    dataloader = DataLoader(
        dataset, batch_size=config.train_batch_size, shuffle=True, drop_last=True)
    model = MLP(
        hidden_size=config.hidden_size,
        hidden_layers=config.hidden_layers,
        emb_size=config.embedding_size,
        time_emb=config.time_embedding,
        input_emb=config.input_embedding,
        device=device)

    noise_scheduler = NoiseScheduler(
        num_timesteps=config.num_timesteps,
        beta_schedule=config.beta_schedule,
        device=device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
    )

    global_step = 0
    frames = []
    residuals = []
    losses = []
    print("Training model...")
    for epoch in range(config.num_epochs):
        avg_loss = 0
        model.train()
        progress_bar = tqdm(total=len(dataloader))
        progress_bar.set_description(f"Epoch {epoch}")
        for step, batch in enumerate(dataloader):
            if config.dataset != 'gaussian_mixture':
                batch = batch[0].to(device)
            noise = torch.randn(batch.shape).to(device)
            timesteps = torch.randint(
                0, noise_scheduler.num_timesteps, (batch.shape[0],)
            ).long()

            noisy = noise_scheduler.add_noise(batch, noise, timesteps)
            noise_pred = model(noisy, timesteps.to(device))
            loss = F.mse_loss(noise_pred, noise)
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "step": global_step}
            avg_loss += loss.detach().item()

            #losses.append(loss.detach().item())
            progress_bar.set_postfix(**logs)
            global_step += 1
        progress_bar.close()
        losses.append(avg_loss / len(dataloader))
        print('Avg train loss for epoch %d: %.4f' % (epoch, avg_loss / len(dataloader)))
        if epoch % config.save_images_step == 0 or epoch == config.num_epochs - 1:
            # generate data with the model to later visualize the learning process
            model.eval()
            sample = torch.randn(config.eval_batch_size, 2).to(device)
            timesteps = list(range(len(noise_scheduler)))[::-1]
            for i, t in enumerate(tqdm(timesteps)):
                t = torch.from_numpy(np.repeat(t, config.eval_batch_size)).long()
                with torch.no_grad():
                    residual = model(sample, t.to(device))
                sample = noise_scheduler.step(residual, t[0], sample)
            frames.append(sample.cpu().numpy())
            residuals.append(residual.cpu().numpy())
    print("Saving model...")
    outdir = f"exps/{config.experiment_name}"
    os.makedirs(outdir, exist_ok=True)

    torch.save({
        "model_state_dict": model.state_dict(),
        "hidden_size": config.hidden_size,
        "hidden_layers": config.hidden_layers,
        "emb_size": config.embedding_size,
        "time_emb": config.time_embedding,
        "input_emb": config.input_embedding
    }, f"{outdir}/model.pth")

    print("Saving images...")
    imgdir = f"{outdir}/images"
    os.makedirs(imgdir, exist_ok=True)
    frames = np.stack(frames)
    xmin, xmax = -1, 1
    ymin, ymax = -1, 1
    for i, frame in enumerate(frames):
        plt.figure(figsize=(10, 10))
        plt.scatter(frame[:, 0], frame[:, 1])
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
        plt.savefig(f"{imgdir}/{i:04}.png")
        plt.close()

    print("Saving loss as numpy array...")
    np.save(f"{outdir}/loss.npy", np.array(losses))

    print("Saving frames...")
    np.save(f"{outdir}/frames.npy", frames)

    print("Saving residuals...")
    np.save(f"{outdir}/residuals.npy", residuals)
