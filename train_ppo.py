import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.optim import Adam
from torch.utils.data import DataLoader
import argparse
from datasets import DataSet
from ddpm import MLP,NoiseScheduler
from discriminator import Discriminator
from ppo import PPO
from train_funs import train_discriminator,train_ppo,plot_2d_data

def kl_prime(x):
    y = 1 + torch.log(torch.clamp(x,min=1e-16))
    if torch.isnan(y).any():
        print('KL Prime is nan')
    return y




def generate_trajectories(noise_scheduler,size,diffusion_net,discriminator,ppo_sampler,device='cuda',gamma=0.99,trajectory_length=2000):
    num_samples,sample_dim = size
    # Single trajectory
    samples = torch.randn(num_samples,sample_dim).to(device)
    timesteps = list(range(len(noise_scheduler)))[::-1]
    print(max(timesteps))
    x_t = {}
    x_t_1 = {}
    f_primes = {}
    actions = {}
    action_probs = {}
    _t = timesteps.pop(0)
    timesteps_=[]
    for i in range(trajectory_length):
        if i == 999:
            print('here')
        samples = samples.to(device)
        t = torch.from_numpy(np.repeat(_t, num_samples)).long().to(device)
        x_t[i] = samples.detach().clone()
        with torch.no_grad():
            residual = diffusion_net(samples, t.to(device))
            density_ratio = discriminator(samples,t.to(device))
            density_ratio = density_ratio / (1-density_ratio+1e-16)
            f_prime = gamma**i*kl_prime(density_ratio)
        action = ppo_sampler.get_action(samples,t.to(device))
        action_prob = ppo_sampler.get_action_prob(samples,t.to(device),action)
        actions[i] = action.detach().clone()
        action_probs[i] = action_prob.detach().clone()
        ode_indices = (actions[i] == 0).squeeze()
        timesteps_.append(_t)
        if ode_indices.any(): # perform one step of ODE
            samples[ode_indices] =noise_scheduler.step(residual[ode_indices], t[0], samples[ode_indices],deterministic=False)
            _t = timesteps.pop(0)
        noise_indices = (actions[i] == 1).squeeze()
        if noise_indices.any():
            noise = torch.randn_like(samples[noise_indices]).to(device)
            times = torch.tensor(1,device=device,dtype=torch.long).repeat(samples[noise_indices].shape[0])
            samples[noise_indices] = noise_scheduler.add_noise(samples[noise_indices],noise,times)
        if i == len(timesteps)-1:
            samples =noise_scheduler.step(residual, t[0], samples,deterministic=True) 
            break
        f_primes[i] = f_prime
        x_t_1[i] = samples.detach().clone()
        if len(timesteps) == 0:
            break
    x_0 = samples.detach().clone()
    return x_t_1,x_t,x_0,timesteps_,f_primes,actions,action_probs





def main():

    parser = argparse.ArgumentParser(description='PPO')
    parser.add_argument('--dataset', type=str, default='Gaussian_2D')
    parser.add_argument('--modes', type=int, default=3)
    parser.add_argument('--num_train_samples', type=int, default=10000)
    parser.add_argument('--num_test_samples', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--epsilon', type=float, default=0.2)
    parser.add_argument('--value_coefficient', type=float, default=0.5)
    parser.add_argument('--entropy_coefficient', type=float, default=0.01)
    parser.add_argument('--clip_grad', type=float, default=0.5)
    parser.add_argument('--pretrained_ckpt', type=str, default='exps/base/model.pth')
    parser.add_argument('--discriminator_ckpt', type=str, default='discriminator.pth')
    parser.add_argument('--n_train_steps', type=int, default=500)
    parser.add_argument('--n_trajectories_epoch', type=int, default=100)
    parser.add_argument('--n_timesteps_buffer', type=int, default=100)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--discriminator_lr', type=float, default=1e-4)
    parser.add_argument('--ppo_hidden_layers', type=int, default=15)
    parser.add_argument('--ppo_emb_size', type=int, default=128)
    parser.add_argument('--ppo_output_dim', type=int, default=2)
    parser.add_argument('--ppo_input_size', type=int, default=2)
    parser.add_argument('--ppo_time_embedding', type=str, default='sinusoidal')
    parser.add_argument('--ppo_lr', type=float, default=1e-4)
    parser.add_argument('--traj_length', type=int, default=2000)
    args = parser.parse_args()
    dataset_type = args.dataset
    print(f'Loading data ---------')
    if dataset_type == 'Gaussian_2D':
        train_dataset = DataSet(modes=args.modes, total_len=args.num_train_samples,remove=False)
        train_dataset.plot(save_path='plots/rl_train_real.png')
        test_dataset = DataSet(modes=args.modes,total_len=args.num_test_samples)
    else:
        raise ValueError('Dataset not supported')
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Loading models -------------------')
    diff_ckpt = torch.load(args.pretrained_ckpt)
    discriminator_ckpt = torch.load(args.discriminator_ckpt)

    # Loading models
    diff_net = MLP(input_emb=diff_ckpt['input_emb'],hidden_size=diff_ckpt['hidden_size'],
                   emb_size=diff_ckpt['emb_size'],time_emb=diff_ckpt['time_emb'],
                  ).to(device)
    
    discriminator_net = Discriminator(n_layers=discriminator_ckpt['n_layers'],hidden_size=discriminator_ckpt['hidden_size'],
                                      emb_size=discriminator_ckpt['emb_size'],time_emb=discriminator_ckpt['time_emb'],
                                      sigmoid=discriminator_ckpt['sigmoid'],input_emb=discriminator_ckpt['input_emb'],input_size=discriminator_ckpt['input_size'],device=device)
    
    diff_net.load_state_dict(diff_ckpt['model_state_dict'])
    discriminator_net.load_state_dict(discriminator_ckpt['model_state_dict'])
    discriminator_optimizer = Adam(discriminator_net.parameters(), lr=args.discriminator_lr)
    noise_scheduler = NoiseScheduler()
    ppo = PPO(emb_size=args.ppo_emb_size,time_embedding=args.ppo_time_embedding,input_size=args.ppo_input_size,hidden_layers=args.ppo_hidden_layers,output_dim=args.ppo_output_dim,device=device)
    ppo_optimizer = Adam(ppo.parameters(), lr=args.ppo_lr)
    ppo_losses = []
    discriminator_losses = []
    # Initial buffer
    print('Generating trajectories -------------------')
    for i in range(args.n_train_steps):
        # Generating data for memory buffer
        fake_train_dataset = diff_net.sample_with_rl(args.num_train_samples,noise_scheduler,ppo,device=device)[-1]
        if i == 0:
            plot_2d_data(fake_train_dataset,f'plots/rl_train_fake_{i}.png')
        fake_train_loader = DataLoader(fake_train_dataset,batch_size=args.batch_size,shuffle=True)
        x_t_1,x_t,x_0,timesteps,f_primes,actions,action_probs = generate_trajectories(noise_scheduler=noise_scheduler,size=(10000,2),diffusion_net=diff_net,ppo_sampler=ppo,discriminator=discriminator_net,device=device,trajectory_length=args.traj_length)
        # Train PPO
        print('Training PPO -------------------')
        ppo_loss = train_ppo(ppo,discriminator_net,x_t,timesteps,f_primes,actions,action_probs,ppo_optimizer,args.epochs,
                  args.n_trajectories_epoch,device=device,epsilon=args.epsilon,n_timesteps_sample=args.n_timesteps_buffer)
        print('PPO Loss ',ppo_loss)
        # Train Discriminator
        print('Training Discriminator -------------------')
        disc_loss = train_discriminator(discriminator_net,discriminator_optimizer,train_loader,fake_train_loader,test_loader,fake_train_loader,noise_scheduler,args.epochs,device,args.discriminator_lr,sigmoid=discriminator_ckpt['sigmoid'])
        print(f'Discriminator Loss {disc_loss}')
        ppo_losses.append(ppo_loss)
        discriminator_losses.append(disc_loss)
        plt.figure(figsize=(10,5))
        plt.subplot(1,2,1)
        plt.plot(ppo_losses)
        plt.title('PPO Loss')
        plt.subplot(1,2,2)
        plt.plot(discriminator_losses)
        plt.title('Discriminator Loss')
        plt.savefig('plots/rl_losses.png')
        plt.close()

        # Save models
        if i % 50 == 0:
            torch.save(discriminator_net.state_dict(), f'exps/base/discriminator_rl_epoch_{i}.pth')
            torch.save(ppo.state_dict(), f'exps/base/ppo_epoch_{i}.pth')
            # Plotting data
            data = fake_train_dataset.detach().cpu().numpy()
            np.save(f'plots/rl_train_{i}.npy',data)
            plt.figure(figsize=(5,5))
            plt.scatter(data[:,0],data[:,1])
            plt.xlim(-5,5)
            plt.ylim(-5,5)
            plt.savefig(f'plots/rl_train_{i}.png')
            plt.show()

        print('Step: ',i)

if __name__ == '__main__':
    main()