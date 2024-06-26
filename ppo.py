import torch
import numpy as np
from positional_embeddings import *


class PPO(torch.nn.Module):

    def __init__(self,
                 emb_size = 128,
                 time_embedding="sinusoidal",
                 input_size = 2,
                 hidden_layers = 5,
                 output_dim = 2,
                 device='cuda'):
        super(PPO, self).__init__()
        self.device = device
        self.time_embedding = PositionalEmbedding(emb_size, time_embedding,device=self.device)
        self.emb_size = emb_size
        self.layers = []
        self.concat_size = input_size + len(self.time_embedding.layer)
        self.layers.append(nn.Linear(self.concat_size, emb_size))
        self.layers.append(nn.ReLU())
        for i in range(hidden_layers):
            self.layers.append(nn.Linear(emb_size, emb_size))
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(emb_size, output_dim))
        self.layers[-1].bias.data = torch.tensor([1.0, 0.0]).to(device)
        self.layers[-1].weight.data = torch.zeros_like(self.layers[-1].weight).to(device)
        self.layers.append(nn.Softmax(dim=-1))
        self.model = nn.Sequential(*self.layers).to(device)



    def forward(self,x_t,t):
        time_emb = self.time_embedding(t).view(t.shape[0],len(self.time_embedding.layer))
        x = torch.cat([x_t,time_emb],dim=-1)
        return self.model(x)
    
    def get_action(self,x_t,t):
        return torch.argmax(self.forward(x_t,t),dim=-1)
    
    def get_action_prob(self,x_t,t,action):
        u = self.forward(x_t,t)
        v = torch.gather(u, 1, action.view(-1,1))
        return v
    

    

    