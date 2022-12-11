import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
# BCQ & BEAR need VAE
class VAE(nn.Module):
    def __init__(
        self,
        input_dim:int,
        output_dim:int,
        hidden_dim:int, # 750 for BCQ & BEAR
        latent_dim:int,
        max_action,
        device,
    ):
        super(VAE,self).__init__()
        self.e1 = nn.Linear(input_dim+output_dim,hidden_dim)
        self.e2 = nn.Linear(hidden_dim,hidden_dim)
        self.mean = nn.Linear(hidden_dim,latent_dim)
        self.log_std = nn.Linear(hidden_dim,latent_dim)
        
        self.d1 = nn.Linear(input_dim+latent_dim,hidden_dim)
        self.d2 = nn.Linear(hidden_dim,hidden_dim)
        self.d3 = nn.Linear(hidden_dim,output_dim)
        
        self.max_action = max_action
        self.latent_dim = latent_dim
        self.device = device
        
    def forward(
        self,
        obs:torch.Tensor,
        action:torch.Tensor,
    ):
        # Encode
        z = F.relu(self.e1(torch.cat([obs,action],1)))
        z = F.relu(self.e2(z))
        
        mean = self.mean(z)
        # Clamped for numerical stability
        log_std = self.log_std(z).clamp(-4,15)
        std = torch.exp(log_std)
        z = mean + std*torch.rand_like(std)
        
        u = self.decode(obs,z)
        return u,mean,std
    
    def decode(
        self,
        obs:torch.Tensor,
        z,
    ):
        # <Ref BCQ the latent vector should be clipped to [-0.5,0.5]>
        if z is None:
            z = torch.randn((obs.shape[0],self.latent_dim)).clamp(-0.5,0.5)
            
        d = F.relu(self.d1(torch.cat([obs,z],1)))
        d = F.relu(self.d2(d))
        return self.max_action * torch.tanh(self.d3(d))
    
    def decode_multiple(self,obs,z=None,num_decode=10):
        if z is None:
            z = torch.FloatTensor(np.random.normal(0,1,size=(obs.shape[0],num_decode,self.latent_dim))).to(self.device)
        a = F.relu(self.d1(torch.cat([obs.unsqueeze(0).repeat(num_decode,1,1).permute(1,0,2),z],2)))
        a = F.relu(self.d2(a))
        return self.max_action*torch.tanh(self.d3(a)),self.d3(a)
        