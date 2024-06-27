import random
from collections import deque
import torch
from train_ppo import kl_prime
class Trajectories:

    def __init__(self,noise_scheduler,state_dim = 2,n_actions =2,gamma=0.99,traj_length=1000,device='cuda'):
        self.trajectories = deque(maxlen=1000)
        self.traj_length = traj_length
        self.device = device
        self.noise_scheduler = noise_scheduler
        self.state_dim = state_dim
        self.n_actions =  n_actions
        self.gamma = gamma



    def __len__(self):
        return len(self.trajectories)
    
    
    def generate(self,n_trajectories,ppo,diff_net,discriminator):
        trajectories = []
        for _ in range(n_trajectories):
            trajectory,done = self.generate_trajectory(ppo,diff_net,discriminator)
            trajectories.append(trajectory)    
        return self.trajectories
    
    def generate_trajectory(self,ppo,diff_net,discriminator):
        sample = torch.randn(1,self.state_dim).to(self.device)
        timesteps = list(range(len(self.noise_scheduler)))[::-1] 
        trajectory = []
        
        for i in range(self.traj_length):
            if len(timesteps) == 0:# if done
                return trajectory,True
            t = torch.tensor(timesteps[0],dtype=torch.float32).to(self.device)
            x_t = sample.detach().clone()
            with torch.no_grad():
                residual = diff_net(sample, t)
                density_ratio = discriminator(sample, t)
                density_ratio = density_ratio / (1 - density_ratio + 1e-16)
                f_prime = self.gamma ** i * kl_prime(density_ratio)

            action = ppo.get_action(sample, t)
            action_prob = ppo.get_action_prob(sample, t, action)
                # Update the samples based on the actions
            if action.item() > 0:
                sample = self.noise_scheduler.step(residual, t, sample, deterministic=True)
                timesteps.pop(0)
            
            else:
                noise = torch.randn_like(sample)
                sample = self.noise_scheduler.add_noise(sample, noise, t)
            trajectory.append((x_t, action, action_prob, f_prime, t))
        
        return trajectory,False

        
