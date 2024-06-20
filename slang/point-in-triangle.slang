import utils;

[AutoPyBindCUDA]
[CUDAKernel]
void run(
    int X, // Acceleration structure length in X
    int Y, // Acceleration structure length in Y
    TensorView<float2> Q, // Array of query point
    TensorView<float2> V, // Array of vertices
    TensorView<int3> T,   // Array of triangles
    TensorView<int> cell_to_triangle_index,
    TensorView<int> cell_to_triangle_index_ptr,
    TensorView<int> QT_idx, // Triangle index for Query point
    TensorView<float3> QT_uv, // Triangle index for Query point
)
{
  uint3 dispatch_id = cudaBlockIdx() * cudaBlockDim() + cudaThreadIdx();
  int qi = dispatch_id.x;
  if (qi >= Q.size(0)) return;

  int NUM_CELLS = cell_to_triangle_index_ptr.size(0);

  float2 q = Q[qi]; // Current query point
  q.x = max(min(q.x, 1.0 - BEPS), BEPS);
  q.y = max(min(q.y, 1.0 - BEPS), BEPS);

  // // First find the cell that the query point belongs to
  int cell_idx = int(q.x * X) * Y + int(q.y * Y);
  cell_idx = min(max(cell_idx, 0), NUM_CELLS - 1);


  // Now linearly search through all the triangles that belong to this cell. For our use case, there should only be a single triangle per cell
  int s = cell_to_triangle_index_ptr[cell_idx];
  int e = cell_to_triangle_index_ptr[cell_idx+1];
  for (int i = s; i < e; i++) {
    int ti = cell_to_triangle_index[i];
    // printf("Query idx %d, Cell idx %d Triangle idx %d\n", qi, cell_idx, ti);
    // if (qi == 0)
    //   printf("Query %f %f, Triangle V0 %f %f, V1 %f %f, V2 %f %f\n", q.x, q.y, V[T[ti].x].x, V[T[ti].x].y, V[T[ti].y].x, V[T[ti].y].y, V[T[ti].z].x, V[T[ti].z].y);
    if (is_inside_triangle(q, V[T[ti].x], V[T[ti].y], V[T[ti].z])) {
      // printf("Inside: Query idx %d, Cell idx %d Triangle idx %d\n", qi, cell_idx, ti);
      QT_idx[qi] = ti;
      QT_uv[qi] = get_barycentrics(q, V[T[ti].x], V[T[ti].y], V[T[ti].z]);
      break;
    }
  }
}