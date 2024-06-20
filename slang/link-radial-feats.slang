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
  int T_NUM,
  TensorView<float2> V, // Array of vertices
  TensorView<int3> T,   // Array of triangles
  TensorView<bool> V_is_continuous,
  TensorView<int> V_adj_disc_idx,
  TensorView<int> V_adj_disc_idx_ptr,
  TensorView<int> V_num_adj_disc_V,
  TensorView<int> T_adj_disc_V,
  TensorView<int> T_adj_disc_feat_idx)
{
  uint3 dispatch_id = cudaBlockIdx() * cudaBlockDim() + cudaThreadIdx();

  if (dispatch_id.x >= T_NUM) return;
  int tri_idx = dispatch_id.x;
  int v_idx = -1;
  for (int k = 0; k < 3; k++) {
    if (k == 0) v_idx = T[tri_idx].x;
    else if (k == 1) v_idx = T[tri_idx].y;
    else v_idx = T[tri_idx].z;
    float2 v = V[v_idx];

    if (V_is_continuous[v_idx]) {
      T_adj_disc_V[tri_idx*6 + k*2 + 0] = -2; // just to mark that it has been visited
      T_adj_disc_V[tri_idx*6 + k*2 + 1] = -2; // just to mark that it has been visited
      T_adj_disc_feat_idx[tri_idx*6 + k*2 + 0] = -2; //just to mark that it has been visited
      T_adj_disc_feat_idx[tri_idx*6 + k*2 + 1] = -2; //just to mark that it has been visited
      continue;
    }

    {
      // CW Direction
      int _v_idx = next_vertex(T[tri_idx], k);
      float2 v_dir = V[_v_idx] - v;
      float min_angle = 10000.0;
      int min_angle_idx = -100;
      int offset_min = -100;
      
      for (int i = 0; i < V_num_adj_disc_V[v_idx]; i++) {
        int vi_idx = get_kth_adj_for_nth_elem(V_adj_disc_idx, V_adj_disc_idx_ptr, i, v_idx);
        float2 vi_dir = V[vi_idx] - v;
        // Measure angle starting from v_i moving toward v in the CCW direction
        float curr_angle = get_angle_ccw(vi_dir, v_dir);
        if (curr_angle < min_angle) {
          min_angle = curr_angle;
          min_angle_idx = vi_idx;
          offset_min = i;
        }
      }
      if (v_idx == 4260 && (tri_idx == 77 || tri_idx == 7597 || tri_idx == 357)) {
        printf("T %d, Next V %d, CW V %d \n", tri_idx, _v_idx, min_angle_idx);
      }
      T_adj_disc_V[tri_idx*6 + k*2 + 0] = min_angle_idx; // nearest disc V idx in CW dir
      T_adj_disc_feat_idx[tri_idx*6 + k*2 + 0] = V_adj_disc_idx_ptr[v_idx] + offset_min; // feature index for CW dir
    };

    {
      // CCW Direction
      int _v_idx = previous_vertex(T[tri_idx], k);
      float2 v_dir = V[_v_idx] - v;
      float min_angle = 10000.0;
      int min_angle_idx = -100;
      int offset_min = -100;
      for (int i = 0; i < V_num_adj_disc_V[v_idx]; i++) {
        int vi_idx = get_kth_adj_for_nth_elem(V_adj_disc_idx, V_adj_disc_idx_ptr, i, v_idx);
        float2 vi_dir = V[vi_idx] - v;
        // Measure angle starting from v moving toward v_i in the CCW direction
        float curr_angle = get_angle_ccw(v_dir, vi_dir);
        // if (tri_idx == 0) {
        //   printf("Proc %d: %d, %f \n", v_idx, vi_idx, curr_angle);
        // }
        if (curr_angle < min_angle) {
          min_angle = curr_angle;
          min_angle_idx = vi_idx;
          offset_min = i;
        }
      }
      if (v_idx == 4260 && (tri_idx == 77 || tri_idx == 7597 || tri_idx == 357)) {
        printf("T %d, Prev V %d, CCW V %d \n", tri_idx, _v_idx, min_angle_idx);
      }
      // if (tri_idx == 0) {
      //   printf("Final %d: %d, %f \n", v_idx, min_angle_idx, min_angle);
      // }
      T_adj_disc_V[tri_idx*6 + k*2 + 1] = min_angle_idx; // nearest disc V idx in CCW dir
      T_adj_disc_feat_idx[tri_idx*6 + k*2 + 1] = V_adj_disc_idx_ptr[v_idx] + offset_min; // feature index for CCW dir
    };
  }
}