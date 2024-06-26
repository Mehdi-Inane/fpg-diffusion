import torch
import matplotlib.pyplot as plt
import numpy as np
from discriminator import Discriminator
from datasets import DataSet
from ddpm import NoiseScheduler,MLP

def make_grid(x, y, h=0.1):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_decision_boundary(model, x, y,save_path='plots/decision_boundary.png'):
    xx, yy = make_grid(x, y)
    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32).to(model.device)
    timesteps = torch.tensor(0, dtype=torch.float).repeat(grid.shape[0]).to(model.device)
    pred = torch.sigmoid(model(grid,timesteps)).detach().cpu().numpy().reshape(xx.shape)
    im = plt.imshow(pred, origin='lower', extent=(-1, 1, -1, 1), aspect='auto', cmap='viridis')
    plt.colorbar(im, label='Discriminator output (probability of real)')
    plt.savefig(save_path)
    plt.show()


def plot_discriminator_output_timesteps(discriminator,denoiser,num_timesteps=50):
    noise_scheduler = NoiseScheduler(num_timesteps=num_timesteps,
                                     beta_schedule='linear',
                                     device=denoiser.device)
    generated_data = denoiser.sample(10000,noise_scheduler)[0]
    real_data = DataSet()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    noise_scheduler = NoiseScheduler(num_timesteps=num_timesteps,
                                     beta_schedule='linear',
                                     device=device)
    timesteps =  list(range(len(noise_scheduler)))[::-1]
    interesting_timesteps = [timesteps[int(len(noise_scheduler)*i/4)] for i in range(0,4)]
    x = np.linspace(-1, 1, 1000)
    y = np.linspace(-1, 1, 1000)
    xx,yy = make_grid(x,y)
    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32).to(device)
    plt.figure(figsize=(60,20))
    for t in interesting_timesteps:
        plt.subplot(3,len(interesting_timesteps),interesting_timesteps.index(t)+1)
        pred = torch.sigmoid(discriminator(grid,torch.tensor(t, dtype=torch.float).repeat(grid.shape[0]).to(device))).detach().cpu().numpy().reshape(xx.shape)
        im = plt.imshow(pred, origin='lower', extent=(-1, 1, -1, 1), aspect='auto', cmap='viridis')
        plt.title(f'Timestep {t}')
    for t in interesting_timesteps:
        plt.subplot(3,len(interesting_timesteps),interesting_timesteps.index(t)+1+len(interesting_timesteps))
        noise_fake = torch.randn(generated_data.shape).to(device)
        pred = noise_scheduler.add_noise(generated_data,noise_fake,torch.tensor(t, dtype=torch.long).repeat(generated_data.shape[0]).to(device)).detach().cpu().numpy()
        plt.scatter(pred[:,0],pred[:,1])
        plt.title(f'Timestep {t}')
    for t in interesting_timesteps:
        plt.subplot(3,len(interesting_timesteps),interesting_timesteps.index(t)+1+2*len(interesting_timesteps))
        noise_real = torch.randn(real_data.data.shape).to(device)
        pred = noise_scheduler.add_noise(real_data.data.to(device),noise_real,torch.tensor(t, dtype=torch.long).repeat(real_data.data.shape[0]).to(device)).detach().cpu().numpy()
        plt.scatter(pred[:,0],pred[:,1])
        plt.title(f'Timestep {t}')
    plt.show()
    plt.savefig(f'plots/noisy_data_timestep.png')





def plot_noising_process(num_timesteps = 50,beta_schedule='linear'):
    real_data = DataSet()
    fake_data = DataSet(remove=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    noise_scheduler = NoiseScheduler(num_timesteps=num_timesteps,
                                     beta_schedule=beta_schedule,
                                     device=device)
    real_noisy = torch.zeros(real_data.data.shape).to(device)
    fake_noisy = torch.zeros(fake_data.data.shape).to(device)
    for i in range(real_data.total_len):
        t = torch.tensor(num_timesteps-1, dtype=torch.long).to(device)
        noise = torch.randn(real_data.data[i].shape).to(device)
        real_noisy[i] = noise_scheduler.add_noise(real_data.data[i].to(device),noise,t)
        fake_noisy[i] = noise_scheduler.add_noise(fake_data.data[i].to(device),noise,t)

    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.scatter(real_noisy[:,0].cpu().numpy(),real_noisy[:,1].cpu().numpy(),label='Real data ')
    plt.xlim(-5,5)
    plt.ylim(-5,5)
    plt.subplot(1,2,2)
    plt.scatter(fake_noisy[:,0].cpu().numpy(),fake_noisy[:,1].cpu().numpy(),label='Fake data ')
    plt.xlim(-5,5)
    plt.ylim(-5,5)
    plt.legend()
    plt.show()
    plt.savefig('plots/noising_process.png')



def compute_kl_divergence(p_samples,p_hat_samples,nbins=50):
    p_samples = p_samples.cpu().numpy()
    p_hat_samples = p_hat_samples.cpu().numpy()
    combined_data = np.vstack((p_samples, p_hat_samples))
    glob_hist,xedges,yedges = np.histogram2d(combined_data[:, 0], combined_data[:, 1], bins=nbins, density=True)
    hist_p, _, _ = np.histogram2d(p_samples[:, 0], p_samples[:, 1], bins=[xedges, yedges], density=True)
    hist_q, _, _ = np.histogram2d(p_hat_samples[:, 0], p_hat_samples[:, 1], bins=[xedges, yedges], density=True)
    hist_q = np.where(hist_q == 0, 1e-10, hist_q)
    hist_p = np.where(hist_p == 0, 1e-10, hist_p)
    hist_p = hist_p/hist_p.sum()
    hist_q = hist_q/hist_q.sum()
    kl = np.sum(np.where((hist_p != 0), hist_p * np.log(hist_p / hist_q), 0))
    
    return kl


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    discriminator_ckpt = torch.load('discriminator.pth')

    model = Discriminator(n_layers=discriminator_ckpt['n_layers'],
                            hidden_size=discriminator_ckpt['hidden_size'],
                            input_size=discriminator_ckpt['input_size'],
                            emb_size=discriminator_ckpt['emb_size'],
                            sigmoid=discriminator_ckpt['sigmoid'],
                            time_emb=discriminator_ckpt['time_emb'],
                            input_emb=discriminator_ckpt['input_emb'],
                            device=device

                          )
    model.load_state_dict(discriminator_ckpt['model_state_dict'])
    x = np.linspace(0, 1, 1000)
    y = np.linspace(0, 1, 1000)
    #plot_decision_boundary(model, x, y)
    #plot_noising_process()
    denoiser_ckpt = torch.load('exps/base/model.pth')
    denoiser = MLP(
        hidden_size=denoiser_ckpt['hidden_size'],
        hidden_layers=denoiser_ckpt['hidden_layers'],
        emb_size=denoiser_ckpt['emb_size'],
        time_emb=denoiser_ckpt['time_emb'],
        input_emb=denoiser_ckpt['input_emb'],
        device=device)
    denoiser.load_state_dict(denoiser_ckpt['model_state_dict']) 
    noise_scheduler = NoiseScheduler(num_timesteps=10,
                                     beta_schedule='linear',
                                     device=device)
    plot_discriminator_output_timesteps(model,denoiser)
    samples_deterministic = denoiser.sample(100000,noise_scheduler,deterministic=True)[0]
    plt.figure(figsize=(10,5))
    plt.subplot(2,2,1)

    plt.scatter(samples_deterministic[:,0].cpu().numpy(),samples_deterministic[:,1].cpu().numpy())
    samples_stochastic = denoiser.sample(100000,noise_scheduler)[0]
    plt.xlim(-2,2)
    plt.ylim(-2,2)
    plt.title('Deterministic samples')
    plt.subplot(2,2,2)
    plt.scatter(samples_stochastic[:,0].cpu().numpy(),samples_stochastic[:,1].cpu().numpy())
    plt.xlim(-2,2)
    plt.ylim(-2,2)
    plt.title('Stochastic samples')
    plt.subplot(2,2,3)
    samples_restart= denoiser.restart_sampling(100000,noise_scheduler,K=8)[0]
    plt.scatter(samples_restart[:,0].cpu().numpy(),samples_restart[:,1].cpu().numpy())
    plt.xlim(-2,2)
    plt.ylim(-2,2)
    plt.title('Restart samples')
    real_samples = DataSet(total_len=100000,remove=False).data
    plt.subplot(2,2,4)
    plt.scatter(real_samples[:,0].cpu().numpy(),real_samples[:,1].cpu().numpy())
    plt.xlim(-2,2)
    plt.ylim(-2,2)
    plt.title('Real samples')
    plt.show()
    plt.savefig('plots/sde_vs_ode.png')
    
    print(f' KL(SDE||p) = {compute_kl_divergence(samples_stochastic,real_samples)}')
    print(f'KL(ODE||p) = {compute_kl_divergence(samples_deterministic,real_samples)}')
    print(f'KL(restart||p) = {compute_kl_divergence(samples_restart,real_samples)}')

if __name__ == "__main__":
    main()