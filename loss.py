import torch



class PPOLoss:

    def __init__(self,actor,critic, clip_ratio, value_coef, entropy_coef,gamma,epsilon = 1e-8,device='cuda'):
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.actor = actor
        self.critic = critic
        self.epsilon = epsilon
        self.gamma = gamma
        self.device = device


    def __call__(self,data):
        obs,actions,f_primes,weights = data['obs'],data['actions'],data['f_primes'],data['weights']
        ratio = self.actor(obs,actions)/self.critic(obs,actions)
        clipped_ratio = torch.clamp(ratio,1-self.clip_ratio,1+self.clip_ratio)
        F = torch.sum(weights*f_primes,dim=1)
        reward = torch.min(ratio*F,clipped_ratio*F).mean()
        return reward

        
