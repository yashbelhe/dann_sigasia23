import torch
import matplotlib.pyplot as plt

def make_comparison_plot(XC, YC, L, res, model, sampler, path):
  _X, _Y = torch.linspace(XC-L/2,XC+L/2,res).cuda(), torch.linspace(YC-L/2,YC+L/2,res).cuda()
  XX, YY = torch.meshgrid(_X, _Y)
  Q = torch.stack([XX.reshape(-1), YY.reshape(-1)], dim=-1)
  gt = sampler(Q=Q)
  plt.figure(figsize=(10,5))
  plt.subplot(1,2,1)
  plt.imshow(gt.detach().cpu().reshape(res,res,3), origin="lower", vmin=0, vmax=1)
  plt.title('Reference')
  plt.axis('off')

  plt.subplot(1,2,2)
  pred = model.forward(Q=Q)
  plt.imshow(pred.detach().cpu().reshape(res,res,3), origin="lower", vmin=0, vmax=1)
  plt.title('Ours')
  plt.axis('off')
  
  plt.savefig(path)
  plt.close()