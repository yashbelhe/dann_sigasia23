import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import torch
import torch.nn as nn
import slangpy
import numpy as np
import matplotlib.pyplot as plt

BLOCK_SIZE_1D=256
def launch_1d(x, LEN):
  x.launchRaw(
  blockSize=(BLOCK_SIZE_1D, 1, 1),
  gridSize=(LEN // BLOCK_SIZE_1D + 1, 1, 1)
)

CUDA_DEVICE = torch.device('cuda')

############# Load all slang modules #############
slang_vertex_is_continuous     = slangpy.loadModule("slang/vertex-is-continuous.slang")
slang_set_curved_inside_sign   = slangpy.loadModule("slang/set-curved-inside-sign.slang")
slang_count_adj_disc           = slangpy.loadModule("slang/count-adj-disc.slang")
slang_add_adj_disc             = slangpy.loadModule("slang/add-adj-disc.slang")
slang_link_radial_feats        = slangpy.loadModule("slang/link-radial-feats.slang")
slang_seco_link_radial_feats   = slangpy.loadModule("slang/seco-link-radial-feats.slang")
slang_count_triangles_per_cell = slangpy.loadModule("slang/count-triangles-per-cell.slang")
slang_add_triangles_per_cell   = slangpy.loadModule("slang/add-triangles-per-cell.slang")
slang_point_in_triangle        = slangpy.loadModule("slang/point-in-triangle.slang")
slang_feature_interpolation    = slangpy.loadModule("slang/feature-interpolation.slang")
slang_d_feature_interpolation  = slangpy.loadModule("slang/d-feature-interpolation.slang")

class DiscontinuityAwareInterpolation(torch.autograd.Function):
  @staticmethod
  def forward(ctx, FEATURE_DIM, Q, V, T, QT_idx, QT_uv, V_is_continuous, T_adj_disc_V, T_adj_disc_feat_idx, V_continuous_feat, V_discontinuous_feat, V_curved_feat, T_bez_cp_idx, T_inside_cubic_sign, seco_T_adj_disc_feat_idx, T_NUM_CURVE):
    Q_NUM = Q.shape[0]
    T_NUM = T.shape[0]
    interpolated_features = torch.zeros((Q_NUM*FEATURE_DIM), dtype=torch.float, device=CUDA_DEVICE)
    launch_1d(slang_feature_interpolation.run(
      T_NUM=T_NUM, FEATURE_DIM=FEATURE_DIM, Q=Q, V=V, T=T,
      QT_idx=QT_idx, QT_uv=QT_uv,
      V_is_continuous=V_is_continuous, 
      V_continuous_feat=V_continuous_feat,
      V_discontinuous_feat=V_discontinuous_feat,
      V_curved_feat=V_curved_feat,
      interpolated_features=interpolated_features,
      T_adj_disc_V=T_adj_disc_V,
      T_adj_disc_feat_idx=T_adj_disc_feat_idx,
      T_bez_cp_idx=T_bez_cp_idx,
      T_inside_cubic_sign=T_inside_cubic_sign,
      seco_T_adj_disc_feat_idx=seco_T_adj_disc_feat_idx,
      T_NUM_CURVE=T_NUM_CURVE
    ), LEN=Q_NUM)
    interpolated_features = interpolated_features.reshape(Q_NUM, FEATURE_DIM)
    ctx.save_for_backward(Q, V, T, QT_idx, QT_uv, V_is_continuous, V_continuous_feat.data, V_discontinuous_feat.data, V_curved_feat.data, T_adj_disc_V, T_adj_disc_feat_idx, interpolated_features, T_bez_cp_idx, T_inside_cubic_sign, seco_T_adj_disc_feat_idx, torch.tensor(T_NUM_CURVE))
    return interpolated_features

  @staticmethod
  def backward(ctx, grad_output):
    (Q, V, T, QT_idx, QT_uv, V_is_continuous, V_continuous_feat, V_discontinuous_feat, V_curved_feat, T_adj_disc_V, T_adj_disc_feat_idx, interpolated_features, T_bez_cp_idx, T_inside_cubic_sign, seco_T_adj_disc_feat_idx, T_NUM_CURVE) = ctx.saved_tensors
    
    FEATURE_DIM = interpolated_features.shape[-1]
    Q_NUM = Q.shape[0]
    T_NUM = T.shape[0]
    
    d_V_continuous_feat    = torch.zeros_like(V_continuous_feat)
    d_V_discontinuous_feat = torch.zeros_like(V_discontinuous_feat)
    d_V_curved_feat        = torch.zeros_like(V_curved_feat)

    d_interpolated_features = grad_output.reshape(-1)
    interpolated_features = interpolated_features.reshape(-1)
    launch_1d(slang_d_feature_interpolation.run(
      T_NUM=T_NUM, FEATURE_DIM=FEATURE_DIM, Q=Q, V=V, T=T,
      QT_idx=QT_idx, QT_uv=QT_uv,
      V_is_continuous=V_is_continuous, 
      V_continuous_feat=V_continuous_feat,
      V_discontinuous_feat=V_discontinuous_feat,
      V_curved_feat=V_curved_feat,
      interpolated_features=interpolated_features,
      T_adj_disc_V=T_adj_disc_V,
      T_adj_disc_feat_idx=T_adj_disc_feat_idx,
      T_bez_cp_idx=T_bez_cp_idx,
      T_inside_cubic_sign=T_inside_cubic_sign,
      seco_T_adj_disc_feat_idx=seco_T_adj_disc_feat_idx,
      T_NUM_CURVE=T_NUM_CURVE,
      d_V_continuous_feat=d_V_continuous_feat,
      d_V_discontinuous_feat=d_V_discontinuous_feat,
      d_V_curved_feat=d_V_curved_feat,
      d_interpolated_features=d_interpolated_features
    ), LEN=Q_NUM)
    
    return tuple([None for _ in range(9)]) + (d_V_continuous_feat, d_V_discontinuous_feat, d_V_curved_feat) + tuple([None for _ in range(4)])

class DANN(torch.nn.Module):
  def __init__(self, mesh, FEATURE_DIM=5, ACCEL_GRID_DIMS=(100, 100), USE_PE=False, OUT_DIM=3, INFERENCE=False):
    super(DANN, self).__init__()
    T_continuous = mesh['continuous_triangles']
    T_linear = mesh['linear_triangles']
    T_curve = mesh['curved_triangles']

    self.T_NUM_CONTINUOUS = len(T_continuous)
    self.T_NUM_LINEAR = len(T_linear)
    self.T_NUM_CURVE = len(T_curve)

    self.T_NUM = self.T_NUM_CURVE + self.T_NUM_LINEAR + self.T_NUM_CONTINUOUS
    self.T_NUM_DISC = self.T_NUM_LINEAR + self.T_NUM_CURVE

    self.T = []
    T_bez_cp_idx = []
    if self.T_NUM_CURVE > 0:
      self.T.append(T_curve[:,:3])
      T_bez_cp_idx = T_curve[:,3:5]
    if self.T_NUM_LINEAR > 0:
      self.T.append(T_linear[:,:3])
    if self.T_NUM_CONTINUOUS > 0:
      self.T.append(T_continuous)

    self.T = torch.tensor(np.concatenate(self.T)).cuda().contiguous()
    self.V = torch.tensor(mesh['vertices']).cuda().contiguous()

    V_NUM = len(self.V)

    # if INFERENCE:
    #   # In inference mode, all the other necessary tensors are already saved in the checkpoint
    #   return

    # For the curved triangles, store the vertex indices for the bezier control points
    if self.T_NUM_CURVE > 0:
      self.register_buffer("T_bez_cp_idx", torch.tensor(T_bez_cp_idx, dtype=torch.int32).cuda().contiguous())
      self.register_buffer("T_inside_cubic_sign", torch.zeros(self.T_NUM_CURVE, dtype=torch.bool).cuda())
    else:
      self.register_buffer("T_bez_cp_idx", torch.tensor([[0, 0]], dtype=torch.int32).cuda())
      self.register_buffer("T_inside_cubic_sign", torch.tensor([0.0], dtype=torch.bool).cuda())
    launch_1d(slang_set_curved_inside_sign.run(T_NUM_CURVE=self.T_NUM_CURVE, V=self.V, T=self.T, T_bez_cp_idx=self.T_bez_cp_idx, T_inside_cubic_sign=self.T_inside_cubic_sign), LEN=self.T_NUM_CURVE)

    # For each vertex is it continuous or not
    # Need to use int because interlockedand is only for int/ float
    temp_V_is_continuous = torch.ones(V_NUM, dtype=torch.int).cuda()
    launch_1d(slang_vertex_is_continuous.run(T_NUM_DISC=self.T_NUM_DISC, V=self.V, T=self.T, V_is_continuous=temp_V_is_continuous), LEN=self.T_NUM_DISC)
    # self.V_is_continuous = temp_V_is_continuous > 0
    self.register_buffer("V_is_continuous", temp_V_is_continuous > 0)

    # Counter Clock Wise (CCW) feature definition: A vertex-edge feature X->Y (for the vertex X) is called counter clockwise if it lies along the directed edge X->Y (i.e the edge appears in a triangle [X,Y,Z] or [Z,X,Y] or [Y,Z,X]).
    # Clock Wise (CW) feature definition: A vertex-edge feature X->Y (for the vertex X) is called clockwise if it is not counter clock wise.

    # Total number of adjcent vertices that are discontinuous
    V_num_adj_disc_V  = torch.zeros(V_NUM, dtype=torch.int).cuda()
    launch_1d(slang_count_adj_disc.run(T_NUM=self.T_NUM, T_NUM_DISC=self.T_NUM_DISC, V=self.V, T=self.T, V_num_adj_disc_V=V_num_adj_disc_V, V_is_continuous=self.V_is_continuous), LEN=self.T_NUM)
    self.V_num_adj_disc_V = V_num_adj_disc_V


    temp_V_num_adj_disc_V = torch.zeros(V_NUM, dtype=torch.int).cuda()

    V_NUM_DIRECTED_EDGES = torch.sum(V_num_adj_disc_V)

    V_adj_disc_idx_ptr = torch.cumsum(V_num_adj_disc_V, dim=0, dtype=torch.int)
    V_adj_disc_idx_ptr -= V_num_adj_disc_V

    V_adj_disc_idx = torch.zeros(torch.sum(V_num_adj_disc_V), dtype=torch.int).cuda() - 1
    launch_1d(slang_add_adj_disc.run(
      T_NUM=self.T_NUM, T_NUM_DISC=self.T_NUM_DISC, V=self.V, T=self.T, 
      temp_V_num_adj_disc_V=temp_V_num_adj_disc_V,
      V_adj_disc_idx_ptr=V_adj_disc_idx_ptr,
      V_adj_disc_idx=V_adj_disc_idx,
      V_is_continuous=self.V_is_continuous
      ), LEN=self.T_NUM)

    self.register_buffer("T_adj_disc_V", torch.zeros((self.T_NUM * 6), dtype=torch.int).cuda() - 1)
    self.register_buffer("T_adj_disc_feat_idx", torch.zeros((self.T_NUM * 6), dtype=torch.int).cuda() - 1)


    launch_1d(slang_link_radial_feats.run(
      T_NUM=self.T_NUM, V=self.V, T=self.T,
      V_is_continuous=self.V_is_continuous,
      V_adj_disc_idx=V_adj_disc_idx,
      V_adj_disc_idx_ptr=V_adj_disc_idx_ptr,
      V_num_adj_disc_V=V_num_adj_disc_V,
      T_adj_disc_V=self.T_adj_disc_V,
      T_adj_disc_feat_idx=self.T_adj_disc_feat_idx,
      ), LEN=self.T_NUM)
    
    self.register_buffer("seco_T_adj_disc_feat_idx", torch.zeros(max((self.T_NUM_CURVE * 2), 1), dtype=torch.int).cuda() - 1)
    launch_1d(slang_seco_link_radial_feats.run(
      T_NUM_CURVE=self.T_NUM_CURVE, V=self.V, T=self.T,
      V_is_continuous=self.V_is_continuous,
      V_adj_disc_idx=V_adj_disc_idx,
      V_adj_disc_idx_ptr=V_adj_disc_idx_ptr,
      V_num_adj_disc_V=V_num_adj_disc_V,
      seco_T_adj_disc_feat_idx=self.seco_T_adj_disc_feat_idx,
      ), LEN=self.T_NUM_CURVE)

    ############### SAVE ALL THE BUFFERS NEEDED FOR INFERENCE ##################
    # self.register_buffer("V_is_continuous",          V_is_continuous)
    # self.register_buffer("T_bez_cp_idx",             T_bez_cp_idx)
    # self.register_buffer("T_inside_cubic_sign",      T_inside_cubic_sign)
    # self.register_buffer("T_adj_disc_V",             T_adj_disc_V)
    # self.register_buffer("T_adj_disc_feat_idx",      T_adj_disc_feat_idx)
    # self.register_buffer("seco_T_adj_disc_feat_idx", seco_T_adj_disc_feat_idx)


    ############### INIITIALIZE ALL FEATURE BUFFERS ##############
    # feature at continuous vertex (memory here can be reduced if we only define these for cont vertex)
    # will require another pointer array to the offsets into this though, which is annoying and has
    # memory requirements too
    self.FEATURE_DIM = FEATURE_DIM
    UR = 0.002
    self.V_continuous_feat    = torch.nn.Parameter(torch.randn((V_NUM * self.FEATURE_DIM), dtype=torch.float, device=CUDA_DEVICE))
    nn.init.uniform_(self.V_continuous_feat.data, -UR, UR)
    # feature at discontinuous vertex. even index: CCW feature, odd index: CW feature
    self.V_discontinuous_feat = torch.nn.Parameter(torch.randn((V_NUM_DIRECTED_EDGES*2 * self.FEATURE_DIM), dtype=torch.float, device=CUDA_DEVICE))
    nn.init.uniform_(self.V_discontinuous_feat.data, -UR, UR)
    # extra feature for single vertex on curved triangle
    self.V_curved_feat        = torch.nn.Parameter(torch.randn((max(self.T_NUM_CURVE, 1) * self.FEATURE_DIM), dtype=torch.float, device=CUDA_DEVICE))
    nn.init.uniform_(self.V_curved_feat.data, -UR, UR)

    MLP_INP_DIM = FEATURE_DIM

    self.mlp = nn.Sequential(
      nn.Linear(MLP_INP_DIM, 64),
      nn.ReLU(),
      nn.Linear(64, 64),
      nn.ReLU(),
      nn.Linear(64, OUT_DIM)
    ).cuda()


    ############# SETUP ACCELERATION STRUCTURE FOR POINT IN TRIANGLE QUERY #############
    self.A_X, self.A_Y = ACCEL_GRID_DIMS
    self.A_NUM_CELLS = self.A_X * self.A_Y
    cell_triangle_count = torch.zeros((self.A_Y, self.A_X), dtype=torch.int, device=CUDA_DEVICE).reshape(-1)

    ######### COUNT THE NUMBER OF TRIANGLES IN EACH CELL ##########
    launch_1d(slang_count_triangles_per_cell.run(Y=self.A_Y, X=self.A_X, T_NUM=self.T_NUM, V=self.V, T=self.T, cell_triangle_count=cell_triangle_count), LEN=self.T_NUM)

    CELL_TOTAL_TRIANGLES = torch.sum(cell_triangle_count)

    # This 1D array has a list of triangles for each cell
    self.cell_to_triangle_index = torch.zeros(CELL_TOTAL_TRIANGLES, dtype=torch.int, device=CUDA_DEVICE) -1
    # This 1D array tells you the index of the first triangle for each cell in `cell_to_triangle_index`
    self.cell_to_triangle_index_ptr = torch.cumsum(cell_triangle_count, dim=0, dtype=torch.int)
    self.cell_to_triangle_index_ptr -= cell_triangle_count
    self.cell_to_triangle_index_ptr = torch.cat([self.cell_to_triangle_index_ptr, torch.tensor([CELL_TOTAL_TRIANGLES], dtype=torch.int, device=CUDA_DEVICE)])
    temp_cell_triangle_count = torch.zeros((self.A_Y, self.A_X), dtype=torch.int, device=CUDA_DEVICE).reshape(-1)

    ########## ADD TRIANGLE IDS TO EACH CELL BUFFER ######
    launch_1d(slang_add_triangles_per_cell.run(Y=self.A_Y, X=self.A_X, T_NUM=self.T_NUM, V=self.V, T=self.T,
      cell_to_triangle_index_ptr=self.cell_to_triangle_index_ptr,
      cell_to_triangle_index=self.cell_to_triangle_index,
      temp_cell_triangle_count=temp_cell_triangle_count,
    ), LEN=self.T_NUM)
  
  '''
  Inputs:  Q: [N,2] array of query points. Every point is in [0,1]^2
  Outputs: [N, out_dims] array of colors/ function values predicted for the query points
  '''
  def forward(self, Q):
    # Step 1: Perform point in triangle query to get triangle indices and barycentric coordinates
    Q_NUM = Q.shape[0]
    # Initialize buffers to store output
    QT_idx = torch.zeros(Q_NUM, dtype=torch.int, device=CUDA_DEVICE) - 1
    QT_uv = torch.zeros((Q_NUM,3), dtype=torch.float, device=CUDA_DEVICE) - 1.0

    launch_1d(slang_point_in_triangle.run(Y=self.A_Y, X=self.A_X, Q=Q, V=self.V, T=self.T,
      cell_to_triangle_index=self.cell_to_triangle_index,
      cell_to_triangle_index_ptr=self.cell_to_triangle_index_ptr,
      QT_idx=QT_idx, QT_uv=QT_uv
    ), LEN=Q_NUM)
    assert torch.all(QT_idx >= 0), "PIT: didn't find a triangle"
    assert torch.all(QT_uv >= 0) and torch.all(QT_uv <= 1), "PIT: UV out of bounds"
    assert torch.allclose(QT_uv.sum(axis=-1), torch.ones_like(QT_uv.sum(axis=-1))), "PIT: UV sum not one"

    interpolated_features = DiscontinuityAwareInterpolation.apply(self.FEATURE_DIM, Q, self.V, self.T, QT_idx, QT_uv, self.V_is_continuous, self.T_adj_disc_V, self.T_adj_disc_feat_idx, self.V_continuous_feat, self.V_discontinuous_feat, self.V_curved_feat, self.T_bez_cp_idx, self.T_inside_cubic_sign, self.seco_T_adj_disc_feat_idx, self.T_NUM_CURVE)

    res = self.mlp(interpolated_features).type(torch.float32)
    return res
