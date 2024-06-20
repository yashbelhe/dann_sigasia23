import utils;
// This routine finds the nearest vertex index in the CCW and CW directions (the last two arguments) that are adjacent to a given vertex (second argument) for a given triangle (first argument) that the vertex is a part of
[AutoPyBindCUDA]
[CUDAKernel]
void run(
  int T_NUM_CURVE,
  TensorView<float2> V, // Array of vertices
  TensorView<int3> T,
  TensorView<int2> T_bez_cp_idx,
  TensorView<bool> T_inside_cubic_sign,
)
{
  uint3 dispatch_id = cudaBlockIdx() * cudaBlockDim() + cudaThreadIdx();
  if (dispatch_id.x >= T_NUM_CURVE) return;
  int tri_idx = dispatch_id.x;
  int3 T_curr = T[tri_idx];
  int2 T_bez_curr = T_bez_cp_idx[tri_idx];

  float2 v0 = V[T_curr.x], v3 = V[T_curr.y], q = V[T_curr.z];
  float2 v1 = l2b(v0, V[T_bez_curr.x], V[T_bez_curr.y], v3, 1),
         v2 = l2b(v0, V[T_bez_curr.x], V[T_bez_curr.y], v3, 2);
  T_inside_cubic_sign[tri_idx] = implicit_cubic(q, v0, v1, v2, v3) > 0.0;
}