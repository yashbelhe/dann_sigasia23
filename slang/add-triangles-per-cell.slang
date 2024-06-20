import utils;

[AutoPyBindCUDA]
[CUDAKernel]
void run(
    int X, // Acceleration structure length in X
    int Y, // Acceleration structure length in Y
    int T_NUM, // Number of triangles
    TensorView<float2> V,                       // Array of vertices
    TensorView<int3> T,                         // Array of triangles
    TensorView<int> cell_to_triangle_index_ptr, // Pointer to next array for each cell
    TensorView<int> cell_to_triangle_index,     // Triangle indinces within each cell
    TensorView<int> temp_cell_triangle_count // Temp counter to avoid deadlock
)
{
    uint3 dispatch_id = cudaBlockIdx() * cudaBlockDim() + cudaThreadIdx();

    if (dispatch_id.x >= T_NUM) return;

    int tri_idx = dispatch_id.x;
    // Get triangle vertices and bounding box
    float2 v0 = V[T[tri_idx].x], v1 = V[T[tri_idx].y], v2 = V[T[tri_idx].z];
    BBox bbox = get_bounding_box(v0, v1, v2, X, Y);


    for (int x = bbox.min_xy_idx.x - 1; x <= bbox.max_xy_idx.x + 1; x++) {
        for (int y = bbox.min_xy_idx.y - 1; y <= bbox.max_xy_idx.y + 1; y++) {
            if (x < 0 || x >= X || y < 0 || y >= Y) continue;
            int oldVal; // Slang just needs this temp variable for interlocked add
            temp_cell_triangle_count.InterlockedAdd(x * Y + y, 1, oldVal);
            int start = cell_to_triangle_index_ptr[x * Y + y];
            // There is guaranteed to be no race condition here
            cell_to_triangle_index[start + oldVal] = tri_idx;
        }
    }
}