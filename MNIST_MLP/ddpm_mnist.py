"""Matched unconditional MNIST experiments. Run --all to compare all four.
Example: .venv/bin/python ddpm_mnist.py --all --steps 3000 --samples 256
Full study: --steps 30000 --samples 10000 --seed 2026 (repeat other seeds).
Shared network/training/evaluation live here so only two model files are needed.
"""
from __future__ import annotations
import argparse
import copy
import json
import math
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.datasets import MNIST
from torchvision.utils import save_image

ROOT = Path(__file__).resolve().parent

class Block(nn.Module):
    def __init__(self, ci, co):
        super().__init__()
        self.n1, self.n2 = nn.GroupNorm(8, ci), nn.GroupNorm(8, co)
        self.c1, self.c2 = nn.Conv2d(ci, co, 3, padding=1), nn.Conv2d(co, co, 3, padding=1)
        self.time = nn.Linear(128, co)
        self.skip = nn.Identity() if ci == co else nn.Conv2d(ci, co, 1)
    def forward(self, x, t):
        h = self.c1(F.silu(self.n1(x))) + self.time(F.silu(t))[:, :, None, None]
        return self.skip(x) + self.c2(F.silu(self.n2(h)))

class UNet(nn.Module):
    def __init__(self, width=32):
        super().__init__()
        w = width
        self.register_buffer('freq', 10000 ** (-torch.arange(32).float() / 31))
        self.temb = nn.Sequential(nn.Linear(64, 128), nn.SiLU(), nn.Linear(128, 128))
        self.input = nn.Conv2d(1, w, 3, padding=1)
        self.enc1 = nn.ModuleList([Block(w,w), Block(w,w)])
        self.down1 = nn.Conv2d(w,2*w,4,2,1)
        self.enc2 = nn.ModuleList([Block(2*w,2*w),Block(2*w,2*w)])
        self.down2 = nn.Conv2d(2*w,4*w,4,2,1)
        self.mid = nn.ModuleList([Block(4*w,4*w),Block(4*w,4*w)])
        self.up2, self.up1 = nn.Conv2d(4*w,2*w,3,padding=1), nn.Conv2d(2*w,w,3,padding=1)
        self.dec2 = nn.ModuleList([Block(4*w,2*w),Block(2*w,2*w)])
        self.dec1 = nn.ModuleList([Block(2*w,w),Block(w,w)])
        self.out = nn.Sequential(nn.GroupNorm(8,w),nn.SiLU(),nn.Conv2d(w,1,3,padding=1))
    def forward(self,x,t):
        a = 1000*t[:,None]*self.freq[None]
        e = self.temb(torch.cat([a.sin(),a.cos()],1))
        h = self.input(x)
        for b in self.enc1: h = b(h,e)
        s1 = h
        h = self.down1(h)
        for b in self.enc2: h = b(h,e)
        s2 = h
        h = self.down2(h)
        for b in self.mid: h = b(h,e)
        h = torch.cat([self.up2(F.interpolate(h,size=(14,14),mode='nearest')),s2],1)
        for b in self.dec2: h = b(h,e)
        h = torch.cat([self.up1(F.interpolate(h,size=(28,28),mode='nearest')),s1],1)
        for b in self.dec1: h = b(h,e)
        return self.out(h)

class DDPM:
    def __init__(self, improved, device, steps=1000):
        self.improved, self.steps = improved, steps
        if improved:
            u = torch.linspace(0,1,steps+1,device=device)
            a = torch.cos((u+.008)/1.008*math.pi/2).square()
            a = a/a[0]
            self.beta = (1-a[1:]/a[:-1]).clamp(max=.999)
        else: self.beta = torch.linspace(1e-4,.02,steps,device=device)
        self.alpha = 1-self.beta
        self.abar = self.alpha.cumprod(0)
        self.prev = torch.cat([torch.ones_like(self.abar[:1]),self.abar[:-1]])
        self.var = self.beta*(1-self.prev)/(1-self.abar)
    def loss(self,model,x):
        i = torch.randint(self.steps,(len(x),),device=x.device)
        a = self.abar[i,None,None,None]
        z = torch.randn_like(x)
        loss = (model(a.sqrt()*x+(1-a).sqrt()*z,(i.float()+1)/self.steps)-z).square().flatten(1).mean(1)
        if self.improved:
            snr = self.abar[i]/(1-self.abar[i])
            loss = loss*snr.clamp(max=5)/snr.clamp_min(1e-8)
        return loss.mean()
    @torch.no_grad()
    def sample(self,model,z,nfe=100):
        x = z.clone()
        for i in reversed(range(self.steps)):
            eps = model(x,torch.full((len(x),),(i+1)/self.steps,device=x.device))
            clean = ((x-(1-self.abar[i]).sqrt()*eps)/self.abar[i].sqrt()).clamp(-1,1)
            x = self.beta[i]*self.prev[i].sqrt()/(1-self.abar[i])*clean + self.alpha[i].sqrt()*(1-self.prev[i])/(1-self.abar[i])*x
            if i: x = x+self.var[i].sqrt()*torch.randn_like(x)
        return (x+1).div(2).clamp(0,1)

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(1,32,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(32,64,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.Flatten(),nn.Linear(64*7*7,64),nn.ReLU())
        self.head = nn.Linear(64,10)
    def forward(self,x):
        f = self.conv(x)
        return self.head(f),f

def sync(device):
    if device.type=='mps': torch.mps.synchronize()
    if device.type=='cuda': torch.cuda.synchronize()

def data():
    tr,te = MNIST(ROOT/'data',train=True,download=True), MNIST(ROOT/'data',train=False,download=True)
    p = torch.randperm(60000,generator=torch.Generator().manual_seed(2026))
    return tr.data[p[:55000],None].float()/255,tr.targets[p[:55000]],tr.data[p[55000:],None].float()/255,tr.targets[p[55000:]],te.data[:,None].float()/255,te.targets

def evaluator(args,device,train,labels,val,vl,test,tl):
    torch.manual_seed(991)
    net = Classifier().to(device)
    opt = torch.optim.Adam(net.parameters(),lr=.001)
    best, state = -1, None
    for epoch in range(args.classifier_epochs):
        net.train()
        for idx in torch.randperm(len(train)).split(256):
            logits,_ = net(train[idx].to(device))
            loss = F.cross_entropy(logits,labels[idx].to(device))
            opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            pred = torch.cat([net(x.to(device))[0].argmax(1).cpu() for x in val.split(256)])
        acc = (pred==vl).float().mean().item()
        print(f'classifier epoch {epoch+1}: validation accuracy {acc:.4f}',flush=True)
        if acc>best: best,state=acc,copy.deepcopy(net.state_dict())
    net.load_state_dict(state); net.eval()
    with torch.no_grad():
        logits,features = zip(*[tuple(y.cpu() for y in net(x.to(device))) for x in test.split(256)])
    accuracy = (torch.cat(logits).argmax(1)==tl).float().mean().item()
    torch.save({'state':state,'validation_accuracy':best,'test_accuracy':accuracy},args.output/'classifier.pt')
    return net,torch.cat(features).numpy(),accuracy

def frechet(real,fake):
    # Symmetric PSD form avoids complex square roots of covariance products.
    a,b=np.cov(real,rowvar=False),np.cov(fake,rowvar=False)
    v,u=np.linalg.eigh(a)
    root=(u*np.sqrt(np.maximum(v,0)))@u.T
    cross=np.linalg.eigvalsh(root@b@root)
    return float(max(0,np.sum((real.mean(0)-fake.mean(0))**2)+np.trace(a)+np.trace(b)-2*np.sqrt(np.maximum(cross,0)).sum()))

def train_run(name,process,args,device,train,classifier,real_features):
    torch.manual_seed(args.seed)
    model = UNet(args.width).to(device)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=0)
    rng = torch.Generator().manual_seed(args.seed)
    start = time.perf_counter(); history=[]
    for step in range(1,args.steps+1):
        idx = torch.randint(len(train),(args.batch_size,),generator=rng)
        x = train[idx].to(device)*2-1
        for group in opt.param_groups: group['lr']=args.lr*min(step/args.warmup,1)
        loss = process.loss(model,x)
        if not torch.isfinite(loss): raise RuntimeError(f'Nonfinite loss in {name}')
        opt.zero_grad(set_to_none=True); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),1); opt.step()
        decay=min(.999,(1+step)/(10+step))
        with torch.no_grad():
            for ep,p in zip(ema.parameters(),model.parameters()): ep.lerp_(p,1-decay)
        history.append(float(loss.detach()))
        if step%100==0 or step==args.steps:
            sync(device)
            print(f'{name}: {step}/{args.steps}, loss={np.mean(history[-100:]):.5f}, seconds={time.perf_counter()-start:.1f}',flush=True)
    sync(device); training_seconds=time.perf_counter()-start
    torch.save({'model':model.state_dict(),'ema':ema.state_dict(),'optimizer':opt.state_dict(),'config':{**vars(args),'output':str(args.output)},'loss':history},args.output/f'{name}.pt')
    torch.manual_seed(args.seed+10000)
    # CPU initial noise is identical across all four experiments.
    z = torch.randn(args.samples,1,28,28)
    images=[]; start=time.perf_counter()
    for batch in z.split(args.sample_batch): images.append(process.sample(ema,batch.to(device),args.rf_nfe).cpu())
    sync(device); seconds=time.perf_counter()-start
    images=torch.cat(images)
    torch.save(images,args.output/f'{name}_samples.pt')
    save_image(images[:100],args.output/f'{name}.png',nrow=10)
    with torch.no_grad():
        out=[tuple(y.cpu() for y in classifier(x.to(device))) for x in images.split(256)]
    logits,features = (torch.cat([x[i] for x in out]) for i in range(2))
    counts=torch.bincount(logits.argmax(1),minlength=10)
    probs=counts.float()/counts.sum()
    result={'name':name,'mnist_feature_frechet':frechet(real_features,features.numpy()),'class_counts':counts.tolist(),'class_entropy':float(-(probs*probs.clamp_min(1e-12).log()).sum()),'mean_classifier_confidence':float(logits.softmax(1).max(1).values.mean()),'training_seconds':training_seconds,'sampling_seconds':seconds,'nfe_per_sample':process.steps if isinstance(process,DDPM) else args.rf_nfe,'samples':args.samples,'seed':args.seed,'optimizer_updates':args.steps,'parameters':sum(p.numel() for p in model.parameters())}
    print(json.dumps(result),flush=True)
    (args.output/f'{name}.json').write_text(json.dumps(result,indent=2))
    return result,images[:100]

def main(family='ddpm'):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--all',action='store_true'); p.add_argument('--variant',choices=['baseline','improved','both'],default='both')
    p.add_argument('--steps',type=int,default=3000); p.add_argument('--batch-size',type=int,default=128)
    p.add_argument('--width',type=int,default=32,choices=[8,16,32,64]); p.add_argument('--lr',type=float,default=2e-4)
    p.add_argument('--warmup',type=int,default=500); p.add_argument('--samples',type=int,default=256)
    p.add_argument('--sample-batch',type=int,default=64); p.add_argument('--rf-nfe',type=int,default=100)
    p.add_argument('--classifier-epochs',type=int,default=3); p.add_argument('--seed',type=int,default=2026)
    p.add_argument('--device',choices=['auto','cpu','mps','cuda'],default='auto'); p.add_argument('--output',type=Path,default=ROOT/'generation_results')
    args=p.parse_args()
    for k in ['steps','batch_size','samples','sample_batch','classifier_epochs','warmup','rf_nfe']:
        if getattr(args,k)<1: p.error(f'{k} must be positive')
    if args.samples<2 or args.rf_nfe%2: p.error('samples must be >=2; rf-nfe must be even for equal Euler/Heun evaluations')
    torch.set_num_threads(4)
    device=torch.device(('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu') if args.device=='auto' else args.device)
    args.output.mkdir(parents=True,exist_ok=True)
    print(f'device={device}; config={vars(args)}',flush=True)
    train,labels,val,vl,test,tl=data()
    clf,features,acc=evaluator(args,device,train,labels,val,vl,test,tl)
    from rectified_flow_mnist import RectifiedFlow
    results=[]; grids=[]
    for fam in (['ddpm','rf'] if args.all else [family]):
        for variant in (['baseline','improved'] if args.variant=='both' else [args.variant]):
            improved=variant=='improved'
            process=DDPM(improved,device) if fam=='ddpm' else RectifiedFlow(improved)
            result,grid=train_run(f'{fam}_{variant}',process,args,device,train,clf,features)
            results.append(result); grids.append(grid)
            (args.output/'comparison.json').write_text(json.dumps({'config':{**vars(args),'output':str(args.output)},'classifier_test_accuracy':acc,'metric_note':'MNIST classifier feature Frechet distance, NOT Inception FID. Lower is better. Single-seed pilot unless repeated. Combined improvements are not isolated ablations.','results':results},indent=2))
    save_image(torch.cat(grids),args.output/'comparison.png',nrow=10)
    ranked=sorted(results,key=lambda r:r['mnist_feature_frechet'])
    lines=['# MNIST generation comparison','',f'Classifier test accuracy: {acc:.2%}. {args.steps} updates per model, seed {args.seed}, {args.samples} generated samples.','', 'Lower MNIST classifier feature Frechet distance is better. This is not Inception FID; small single-seed runs cannot establish a universal winner. Raw losses across model families are not comparable.','', '| Model | Feature distance | Class entropy | NFE | Train seconds | Sample seconds |','|---|---:|---:|---:|---:|---:|']
    for r in ranked: lines.append(f"| {r['name']} | {r['mnist_feature_frechet']:.3f} | {r['class_entropy']:.3f} | {r['nfe_per_sample']} | {r['training_seconds']:.1f} | {r['sampling_seconds']:.1f} |")
    lines+=['','Comparison image: each consecutive block of 100 images follows execution order: '+', '.join(r['name'] for r in results)+'.','', 'Training uses a fixed 55,000/5,000 split, no augmentation, identical architecture, initialization seed and batch indices. Classifier selection uses validation accuracy; the 10,000 official test images supply reference features. DDPM uses the full 1,000-step posterior chain; RF Euler and Heun use equal NFE. EMA decay ramps to 0.999. The improved variants bundle two changes, so these runs cannot attribute gains to either component alone.']
    (args.output/'comparison.md').write_text('\n'.join(lines)+'\n')
    print(f"Results saved to {args.output}. Best measured feature distance: {ranked[0]['name']}",flush=True)

if __name__=='__main__': main()
