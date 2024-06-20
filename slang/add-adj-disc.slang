import utils;

[AutoPyBindCUDA]
[CUDAKernel]
void run(
  int T_NUM,
  int T_NUM_DISC,
  TensorView<float2> V, // Array of vertices
  TensorView<int3> T,   // Array of triangles
  TensorView<int> temp_V_num_adj_disc_V,
  TensorView<int> V_adj_disc_idx_ptr,
  TensorView<int> V_adj_disc_idx,
  TensorView<bool> V_is_continuous
)
{
  uint3 dispatch_id = cudaBlockIdx() * cudaBlockDim() + cudaThreadIdx();

  // Continuous triangle, continue
  if (dispatch_id.x >= T_NUM) return;
  int tri_idx = dispatch_id.x;
  // The first edge [X,Y] is always the discontinuous one
  int x = T[tri_idx].x, y = T[tri_idx].y, z = T[tri_idx].z;

  float2 vx = V[T[tri_idx].x], vy = V[T[tri_idx].y], vz = V[T[tri_idx].z];
  int oldVal;
  bool v0 = !V_is_continuous[T[tri_idx].x];
  bool v1 = !V_is_continuous[T[tri_idx].y];
  bool v2 = !V_is_continuous[T[tri_idx].z];
  if (v0 && v1) {
    if (tri_idx < T_NUM_DISC || (idb(vx) && idb(vy) && dbi(vx) == dbi(vy))) {
      temp_V_num_adj_disc_V.InterlockedAdd(x, 1, oldVal);
      V_adj_disc_idx[V_adj_disc_idx_ptr[x] + oldVal] = y;
      temp_V_num_adj_disc_V.InterlockedAdd(y, 1, oldVal);
      V_adj_disc_idx[V_adj_disc_idx_ptr[y] + oldVal] = x;
    }

  }
  if (v1 && v2) {
    if (idb(vz) && idb(vy) && dbi(vz) == dbi(vy)) {
      temp_V_num_adj_disc_V.InterlockedAdd(y, 1, oldVal);
      V_adj_disc_idx[V_adj_disc_idx_ptr[y] + oldVal] = z;
      temp_V_num_adj_disc_V.InterlockedAdd(z, 1, oldVal);
      V_adj_disc_idx[V_adj_disc_idx_ptr[z] + oldVal] = y;
    }
  }
  if (v0 && v2) {
    if (idb(vz) && idb(vx) && dbi(vz) == dbi(vx)) {
      temp_V_num_adj_disc_V.InterlockedAdd(z, 1, oldVal);
      V_adj_disc_idx[V_adj_disc_idx_ptr[z] + oldVal] = x;
      temp_V_num_adj_disc_V.InterlockedAdd(x, 1, oldVal);
      V_adj_disc_idx[V_adj_disc_idx_ptr[x] + oldVal] = z;
    }
  }
}