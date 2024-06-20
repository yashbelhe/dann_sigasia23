import torch
import numpy as np
import matplotlib.pyplot as plt

import model
import samplers
import os
import utils
import configs
import argparse

parser = argparse.ArgumentParser(description='Optional app description')
parser.add_argument('expt_name', type=str, help='Name of the experiment')
args = parser.parse_args()

torch.manual_seed(30)

if args.expt_name == "flowerpot":
  config = configs.CONFIGS['rendering_flowerpot']
elif args.expt_name == "shapes":
  config = configs.CONFIGS['vg_shapes']
elif args.expt_name == "overview":
  config = configs.CONFIGS['wos_overview']
elif args.expt_name == "circles":
  config = configs.CONFIGS['wos_circles']
else:
  assert False, "unsupported scene"

SAVE_INT = 100
FEATURE_DIM = 5
ACCEL_GRID_DIMS = (2000, 2000)
LR = 0.01
BETAS = (0.9, 0.99)
NUM_QUERIES_SQRT = config['SAMPLING_GRID_SIZE']
NUM_ITERS = config['NUM_ITERS']
BATCH_SIZE = config['BATCH_SIZE']

fname = config['FNAME']
results_dir = os.path.join('results', *fname.split('/')[1:3])
if not os.path.exists(results_dir):
  os.makedirs(results_dir)


app = fname.split('/')[1]

sampler = samplers.BaseSampler()
if app == "vg":
  sampler = samplers.VGSampler(fname)
elif app == "rendering":
  sampler = samplers.RenderingSampler(fname)
elif app == "wos":
  sampler = samplers.WoSSampler(fname)

mesh = np.load(fname)
model = model.DANN(mesh=mesh, FEATURE_DIM=FEATURE_DIM, ACCEL_GRID_DIMS=ACCEL_GRID_DIMS)
optim = torch.optim.Adam(model.parameters(), lr=LR, betas=BETAS)

iter_count = 0
for idx in range(NUM_ITERS):
  if config['SAMPLING'] == 'triangle':
    Q = samplers.get_stratified_in_triangles(model=model)
  elif config['SAMPLING'] == 'grid':
    Q = samplers.get_stratified_random(NUM_QUERIES_SQRT)
  else:
    assert False
  randperm = torch.randperm(Q.shape[0])
  Q = Q[randperm]

  gt = sampler(Q)

  for batch_idx in range((Q.shape[0] + BATCH_SIZE - 1) // BATCH_SIZE):
    Qb = Q[batch_idx*BATCH_SIZE: (batch_idx+1)*BATCH_SIZE]
    gtb = gt[batch_idx*BATCH_SIZE: (batch_idx+1)*BATCH_SIZE]

    predb = model(Q=Qb)
    if app == "rendering":
      loss = ((predb-gtb).square()/(predb.detach().square() + 0.01)).mean() # rendering
    else:
      loss = (predb-gtb).square().mean() # vg, wos

    loss.backward()
    optim.step()
    optim.zero_grad()

    print(f"Iter {iter_count}, Loss: {loss.item()}")
    
    if iter_count % SAVE_INT == 0:
      torch.save({'model': model.state_dict()}, os.path.join(results_dir, f"checkpoint_{iter_count}.pth"))
      locs = [(0.5, 0.5, 1)]
      # locs = [(0.5, 0.5, 1), (0.57695, 0.34957, 0.006), (0.513, 0.4012, 0.005), (0.57195, 0.34757, 0.1)]
      for loc_idx, loc in enumerate(locs):
        utils.make_comparison_plot(loc[0], loc[1], loc[2], 500, model, sampler, os.path.join(results_dir, f"output_{loc_idx}_{iter_count}.png"))

    iter_count += 1
  if iter_count > NUM_ITERS:
    break
