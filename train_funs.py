import torch
import random
import numpy as np
import matplotlib.pyplot as plt


def plot_2d_data(_data,save_path):
    data = _data.cpu().detach().numpy()
    plt.scatter(data[:,0],data[:,1])
    plt.savefig(save_path)
    plt.close()




def train_discriminator(discriminator,optimizer,real_train_loader,fake_train_loader,real_eval_loader,fake_eval_loader,noise_scheduler,n_epochs,device,lr,sigmoid):
    if sigmoid:
        criterion = torch.nn.BCELoss()
    else:
        criterion = torch.nn.BCEWithLogitsLoss()
    avg_losses = []
    eval_losses = []
    for epoch in range(n_epochs):
        discriminator.train()
        avg_loss = 0
        for i, (real_batch, fake_batch) in enumerate(zip(real_train_loader,fake_train_loader)):
            real_batch = real_batch.to(device)
            fake_batch = fake_batch.to(device)
            real_labels = torch.ones(real_batch.size(0),1).to(device)
            fake_labels = torch.zeros(fake_batch.size(0),1).to(device)
            optimizer.zero_grad()
            noise_real = torch.randn(real_batch.shape).to(device)
            noise_fake = torch.randn(fake_batch.shape).to(device)
            timesteps = torch.randint(
                0, noise_scheduler.num_timesteps, (real_batch.shape[0],)
            ).long()

            noisy = noise_scheduler.add_noise(real_batch, noise_real, timesteps)
            real_output = discriminator(noisy, timesteps.to(device))
            timesteps = torch.randint(
                0, noise_scheduler.num_timesteps, (fake_batch.shape[0],)
            ).long()
            noisy = noise_scheduler.add_noise(fake_batch, noise_fake, timesteps)
            fake_output = discriminator(noisy,timesteps.to(device))
            real_loss = criterion(real_output,real_labels)
            fake_loss = criterion(fake_output,fake_labels)
            loss = real_loss + fake_loss
            avg_loss += loss.item()
            loss.backward()
            optimizer.step()
        avg_loss /= len(real_train_loader)
        avg_losses.append(avg_loss)
    return np.mean(avg_losses)


def train_ppo(ppo_net,discriminator_net,x_t,timesteps,f_primes,actions,action_probs,optimizer,
              n_epochs,n_trajectories_epoch,device,epsilon=1e-2,n_timesteps_sample=100):
    n_timesteps = len(timesteps)
    n_trajectories = x_t[0].shape[0]
    avg_losses = []
    for epoch in range(n_epochs):
        ppo_net.train()
        discriminator_net.eval()
        avg_loss = 0
        sampled_indices = torch.randperm(n_trajectories)[:n_trajectories_epoch]
        timesteps_ =[timesteps[i] for i in range(len(timesteps)) if timesteps[i] !=0]
        sampled_timesteps = np.random.choice(timesteps_,n_timesteps_sample,replace=False)
        x_t_i = torch.stack([x_t[t][sampled_indices] for t in sampled_timesteps ]).view(-1,2)
        f_prime = torch.stack([f_primes[t][sampled_indices] for t in sampled_timesteps]).view(-1,1)
        action = torch.stack([actions[t][sampled_indices] for t in sampled_timesteps]).view(-1,1)
        times = torch.stack([torch.tensor(t,device=device,dtype=torch.long).repeat(n_trajectories_epoch) for t in sampled_timesteps]).to(device).view(-1,1)
        optimizer.zero_grad()
        action_prob = ppo_net.get_action_prob(x_t_i,times,action)
        critic_action_prob = torch.stack([action_probs[t][sampled_indices] for t in sampled_timesteps]).reshape(n_timesteps_sample*n_trajectories_epoch,1)
        prob = action_prob/critic_action_prob
        loss = torch.mean(torch.min(prob*f_prime,torch.clamp(prob,1-epsilon,1+epsilon)*f_prime))
        avg_loss += loss.item()
        loss.backward()
        optimizer.step()
        avg_losses.append(loss.item())
        return avg_loss

        


