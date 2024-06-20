import utils;

[AutoPyBindCUDA]
[CUDAKernel]
void run(
  int T_NUM_DISC,
  TensorView<float2> V, // Array of vertices
  TensorView<int3> T,   // Array of triangles
  TensorView<int> V_is_continuous
)
{
  uint3 dispatch_id = cudaBlockIdx() * cudaBlockDim() + cudaThreadIdx();

  // Continuous triangle, continue
  if (dispatch_id.x >= T_NUM_DISC) return;
  int tri_idx = dispatch_id.x;
  // The first edge is always the discontinuous one
  // We don't need to care about atomics since even in a race condition the value set will be false
  int oldVal;
  V_is_continuous.InterlockedAdd(T[tri_idx].x, -1, oldVal);
  V_is_continuous.InterlockedAdd(T[tri_idx].y, -1, oldVal);

  // Vertices on domain boundary are continuous, no discontinuities allowed here
  // if (
  //   V[T[tri_idx].z].x < BEPS       || V[T[tri_idx].z].y < BEPS       ||
  //   V[T[tri_idx].z].x > 1.0 - BEPS || V[T[tri_idx].z].y > 1.0 - BEPS 
  // ) {
  //   V_is_continuous.InterlockedAdd(T[tri_idx].z, -1, oldVal);
  // }

  if (
    V[T[tri_idx].z].x < BEPS       || V[T[tri_idx].z].y < BEPS       ||
    V[T[tri_idx].z].x > 1.0 - BEPS || V[T[tri_idx].z].y > 1.0 - BEPS 
  ) {
    V_is_continuous.InterlockedAdd(T[tri_idx].z, -1, oldVal);
  }
  // V_is_continuous[T[tri_idx].x] = false;
  // V_is_continuous[T[tri_idx].y] = false;
}