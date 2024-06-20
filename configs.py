CONFIGS = {}

CONFIGS['rendering_flowerpot'] = {
  'NUM_ITERS': 2000,
  'BATCH_SIZE': 2**19,
  'LR': 0.01,
  'FNAME': 'data/rendering/flowerpot/img.msh_curved.npz',
  'SAMPLING': 'triangle',
  'SAMPLING_GRID_SIZE': -1 # set to -1 for triangle sampling
}

CONFIGS['vg_shapes'] = {
  'NUM_ITERS': 2000,
  'BATCH_SIZE': 2**19,
  'LR': 0.01,
  'FNAME': 'data/vg/shapes/img.msh_curved.npz',
  'SAMPLING': 'grid',
  'SAMPLING_GRID_SIZE': 10000
}

CONFIGS['wos_circles'] = {
  'NUM_ITERS': 2000,
  'BATCH_SIZE': 2**16,
  'LR': 0.01,
  'FNAME': 'data/wos/circles/img.msh_curved.npz',
  'SAMPLING': 'grid',
  'SAMPLING_GRID_SIZE': 2000
}

CONFIGS['wos_overview'] = {
  'NUM_ITERS': 2000,
  'BATCH_SIZE': 2**16,
  'LR': 0.01,
  'FNAME': 'data/wos/overview/img.msh_curved.npz',
  'SAMPLING': 'grid',
  'SAMPLING_GRID_SIZE': 2000
}