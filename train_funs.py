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




def check_lengths(x_t,a_t,f_prime,action_prob,timesteps):
    assert len(x_t) == len(a_t) == len(f_prime) == len(action_prob) == len(timesteps)
    for i in range(len(x_t)):
        try:
            assert x_t[i].shape[0] == a_t[i].shape[0] == f_prime[i].shape[0] == action_prob[i].shape[0] == timesteps[i].shape[0]
        except:
            print(f"Lengths for x_t, a_t, f_prime, action_prob, timesteps at index {i} are not equal")
            print(f'length of x_t: {x_t[i].shape[0]}')
            print(f'length of a_t: {a_t[i].shape[0]}')
            print(f'length of f_prime: {f_prime[i].shape[0]}')
            print(f'length of action_prob: {action_prob[i].shape[0]}')
            print(f'length of timesteps: {timesteps[i].shape[0]}')
            exit()
            return False
    return True

def sample_trajectory_data(x_t,a_t,f_prime,action_prob,timesteps,traj_length,traj_batch=1000,time_batch=100):
    n_trajectories = a_t[0].shape[0]
    sampled_trajectories = random.sample(list(range((n_trajectories))),time_batch)
    #print(check_lengths(x_t,a_t,f_prime,action_prob,timesteps))
    trajectories = {
        'x_t':torch.stack([x_t[i][sampled_trajectories] for i in range(traj_length)]),
        'action':torch.stack([a_t[i][sampled_trajectories] for i in range(traj_length)]),
        'f_t':torch.stack([f_prime[i][sampled_trajectories] for i in range(traj_length)]),
        'action_probs':torch.stack([action_prob[i][sampled_trajectories] for i in range(traj_length)]),
        'diff_timestep':torch.stack([timesteps[i][sampled_trajectories] for i in range(traj_length)])
    }
    sampled_timesteps = random.sample(range(traj_length), time_batch)

    # Gather data for sampled timesteps
    batch_data = {
        'x_t': torch.stack([trajectories['x_t'][t] for t in sampled_timesteps]),
        'action': torch.stack([trajectories['action'][t] for t in sampled_timesteps]),
        'f_t': torch.stack([trajectories['f_t'][t] for t in sampled_timesteps]),
        'f_t': torch.stack([trajectories['f_t'][t] for t in sampled_timesteps]),
        'action_probs': torch.stack([trajectories['action_probs'][t] for t in sampled_timesteps]),
        'diff_timestep': torch.stack([trajectories['diff_timestep'][t] for t in sampled_timesteps])
    }

    return batch_data, sampled_timesteps,sampled_trajectories




def train_ppo(ppo_net,discriminator_net,x_t,timesteps,f_primes,actions,action_probs,optimizer,
              n_epochs,n_trajectories_epoch,device,epsilon=1e-2,n_timesteps_sample=100,traj_length=2000):
    n_timesteps = len(timesteps)
    n_trajectories = x_t[0].shape[0]
    avg_losses = []
    for epoch in range(n_epochs):
        ppo_net.train()
        discriminator_net.eval()
        avg_loss = 0
        batch,sampled_timesteps,sampled_indices = sample_trajectory_data(x_t,actions,f_primes,action_probs,timesteps,traj_length,n_trajectories_epoch,n_timesteps_sample)
        x_t_i = batch['x_t'].view(-1,2)
        f_prime = batch['f_t'].view(-1,1)
        action = batch['action'].view(-1,1)
        times = batch['diff_timestep'].view(-1,1)
        critic_action_prob = batch['action_probs'].view(-1,1)
        optimizer.zero_grad()
        action_prob = ppo_net.get_action_prob(x_t_i,times,action)
        prob = action_prob/critic_action_prob
        loss = torch.mean(torch.min(prob*f_prime,torch.clamp(prob,1-epsilon,1+epsilon)*f_prime))
        avg_loss += loss.item()
        loss.backward()
        optimizer.step()
        avg_losses.append(loss.item())
        return avg_loss

        


