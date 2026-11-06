"""Unconditional rectified flow: uniform/Euler and logit-normal/Heun.
Run: .venv/bin/python rectified_flow_mnist.py --steps 3000
Shared U-Net, matched training, metrics and CLI are in ddpm_mnist.py.
Time runs from Gaussian noise (0) to clean MNIST (1).
"""
import torch
from torch.nn import functional as F

class RectifiedFlow:
    def __init__(self, improved=False):
        self.improved=improved
    def loss(self,model,clean):
        z=torch.randn_like(clean)
        t=torch.randn(len(clean),device=clean.device).sigmoid() if self.improved else torch.rand(len(clean),device=clean.device)
        s=t[:,None,None,None]
        return F.mse_loss(model((1-s)*z+s*clean,t),clean-z)
    @torch.no_grad()
    def sample(self,model,z,nfe=100):
        if self.improved and nfe%2: raise ValueError('Heun requires an even NFE budget')
        steps=nfe//2 if self.improved else nfe
        x=z.clone(); h=1/steps
        for i in range(steps):
            t=torch.full((len(x),),i/steps,device=x.device)
            v=model(x,t)
            if self.improved:
                next_t=torch.full((len(x),),(i+1)/steps,device=x.device)
                x=x+.5*h*(v+model(x+h*v,next_t))
            else: x=x+h*v
        return ((x+1)/2).clamp(0,1)

if __name__=='__main__':
    from ddpm_mnist import main
    main('rf')
