import utils;

int next_vertex(int3 tri, int v_idx) {
  if (v_idx == 0) return tri.y;
  else if (v_idx == 1) return tri.z;
  else return tri.x;
}

int previous_vertex(int3 tri, int v_idx) {
  if (v_idx == 0) return tri.z;
  else if (v_idx == 1) return tri.x;
  else return tri.y;
}

int get_kth_adj_for_nth_elem(TensorView<int> data, TensorView<int> data_ptr, int k, int n) {
  return data[data_ptr[n] + k];
}

// This routine finds the nearest vertex index in the CCW and CW directions (the last two arguments) that are adjacent to a given vertex (second argument) for a given triangle (first argument) that the vertex is a part of
[AutoPyBindCUDA]
[CUDAKernel]
void run(
    int T_NUM_CURVE,
    TensorView<float2> V, // Array of vertices
    TensorView<int3> T,   // Array of triangles
    TensorView<bool> V_is_continuous,
    TensorView<int> V_adj_disc_idx,
    TensorView<int> V_adj_disc_idx_ptr,
    TensorView<int> V_num_adj_disc_V,
    TensorView<int> seco_T_adj_disc_feat_idx)
{
  uint3 dispatch_id = cudaBlockIdx() * cudaBlockDim() + cudaThreadIdx();

  if (dispatch_id.x >= T_NUM_CURVE) return;
  int tri_idx = dispatch_id.x;
  int v_idx = -1;
  
  // We only need to do this for the first and second vertex along which the curved edge lies
  for (int k = 0; k < 2; k++) {
    if (k == 0) v_idx = T[tri_idx].x;
    else if (k == 1) v_idx = T[tri_idx].y;
    else v_idx = T[tri_idx].z;

    // if (V_is_continuous[v_idx]) {
    //   T_adj_disc_V[tri_idx*6 + k*2 + 0] = -2; // just to mark that it has been visited
    //   T_adj_disc_V[tri_idx * 6 + k * 2 + 1] = -2;             // just to mark that it has been visited
    //   seco_T_adj_disc_feat_idx[tri_idx * 6 + k * 2 + 0] = -2; // just to mark that it has been visited
    //   seco_T_adj_disc_feat_idx[tri_idx*6 + k*2 + 1] = -2; //just to mark that it has been visited
    //   continue;
    // }

    if (k == 0) {
      // CW Direction
      // int vn_idx = next_vertex(T[tri_idx], k);
      int vn_idx = previous_vertex(T[tri_idx], k);
      float2 vn_dir = V[vn_idx] - V[v_idx];
      float vn_min_angle = 10000.0;
      int vn_min_angle_idx = -100;
      int offset_min = -100;
      for (int i = 0; i < V_num_adj_disc_V[v_idx]; i++) {
        int vm_idx = get_kth_adj_for_nth_elem(V_adj_disc_idx, V_adj_disc_idx_ptr, i, v_idx);
        float2 vm_dir = V[vm_idx] - V[v_idx];
        float curr_angle = get_angle_ccw(vn_dir, vm_dir);
        if (curr_angle < vn_min_angle) {
          vn_min_angle = curr_angle;
          vn_min_angle_idx = vm_idx;
          offset_min = i;
        }
      }
      seco_T_adj_disc_feat_idx[tri_idx*2 + 0] = V_adj_disc_idx_ptr[v_idx] + offset_min; // feature index for CW dir
    };

    if (k == 1) {
      // CCW Direction
      int vp_idx = next_vertex(T[tri_idx], k);
      // int vp_idx = previous_vertex(T[tri_idx], v_idx);
      float2 vp_dir = V[vp_idx] - V[v_idx];
      float vp_min_angle = 10000.0;
      int vp_min_angle_idx = -100;
      int offset_min = -100;
      for (int i = 0; i < V_num_adj_disc_V[v_idx]; i++) {
        int vm_idx = get_kth_adj_for_nth_elem(V_adj_disc_idx, V_adj_disc_idx_ptr, i, v_idx);
        float2 vm_dir = V[vm_idx] - V[v_idx];
        float curr_angle = get_angle_ccw(vm_dir, vp_dir);
        if (curr_angle < vp_min_angle) {
          vp_min_angle = curr_angle;
          vp_min_angle_idx = vm_idx;
          offset_min = i;
        }
      }
      seco_T_adj_disc_feat_idx[tri_idx*2 + 1] = V_adj_disc_idx_ptr[v_idx] + offset_min; // feature index for CCW dir
    };
  }
}