import utils;
#define EPS_DIV 0.000000001

// This routine finds the nearest vertex index in the CCW and CW directions (the last two arguments) that are adjacent to a given vertex (second argument) for a given triangle (first argument) that the vertex is a part of
[AutoPyBindCUDA]
[CUDAKernel]
void run(
  int T_NUM,
  int T_NUM_CURVE,
  int FEATURE_DIM,
  TensorView<float2> V, // Array of vertices
  TensorView<int3> T,   // Array of triangles
  TensorView<float2> Q, // Array of query points
  TensorView<int> QT_idx,   // Array of triangles indices that contain the respective query points
  TensorView<float3> QT_uv, // Array of barycentrics for each query point
  TensorView<bool> V_is_continuous,
  TensorView<int2> T_bez_cp_idx,
  TensorView<bool> T_inside_cubic_sign,
  TensorView<int> T_adj_disc_V,
  TensorView<int> T_adj_disc_feat_idx,
  TensorView<int> seco_T_adj_disc_feat_idx,
  TensorView<float> V_continuous_feat,
  TensorView<float> V_discontinuous_feat,
  TensorView<float> V_curved_feat,
  TensorView<float> interpolated_features
)
{
  uint3 dispatch_id = cudaBlockIdx() * cudaBlockDim() + cudaThreadIdx();

  if (dispatch_id.x >= Q.size(0)) return;
  int q_idx = dispatch_id.x; // Current query index
  int tri_idx = QT_idx[q_idx]; // Triangle index for the current query point
  int3 T_curr = T[tri_idx];    // Triangle
  int2 T_bez_curr = T_bez_cp_idx[tri_idx];    // Triangle
  float3 uv = QT_uv[q_idx]; // Barycentric
  
  int v_idx = -1; // Vertex index within the triangle
  float2 q = Q[q_idx]; // query location
  float2 v = float2(-1); // vertex location

  bool is_in_curved_region = false;

  // Is a curved triangle
  if (tri_idx < T_NUM_CURVE) {
    // Is inside the curved region inside the curved triangle
    float2 v0 = V[T_curr.x], v3 = V[T_curr.y];
    float2 v1 = l2b(v0, V[T_bez_curr.x], V[T_bez_curr.y], v3, 1),
           v2 = l2b(v0, V[T_bez_curr.x], V[T_bez_curr.y], v3, 2);
    bool cubic_sign = implicit_cubic(q, v0, v1, v2, v3) > 0.0;
    if (cubic_sign != T_inside_cubic_sign[tri_idx]) {
      is_in_curved_region = true;
    }
  }

  for (int k = 0; k < 3; k++) {
    float uv_curr = 0.0; // barycentric coordinate for current vertex query point pair
    if (k == 0) {
      v_idx = T_curr.x;
      uv_curr = uv.x;
    } else if (k == 1) {
      v_idx = T_curr.y;
      uv_curr = uv.y;
    }
    else {
      v_idx = T_curr.z;
      uv_curr = uv.z;
    }
    v = V[v_idx];

    if (is_in_curved_region) {
      int offset_v0 = seco_T_adj_disc_feat_idx[tri_idx * 2 + 0];
      int offset_v1 = seco_T_adj_disc_feat_idx[tri_idx * 2 + 1];

      for (int f = 0; f < FEATURE_DIM; f++) {
        if (k == 0) {
          interpolated_features[FEATURE_DIM*q_idx + f] += uv_curr*V_discontinuous_feat[2*FEATURE_DIM*offset_v0 + FEATURE_DIM + f];
        } else if (k == 1) {
          interpolated_features[FEATURE_DIM*q_idx + f] += uv_curr*V_discontinuous_feat[2*FEATURE_DIM*offset_v1 + f];
        } else if (k == 2) {
          interpolated_features[FEATURE_DIM*q_idx + f] += uv_curr*V_curved_feat[FEATURE_DIM*tri_idx + f];
        }
      }
    } else {
      if (V_is_continuous[v_idx]) {
        // Continuous vertex, no interpolation needed
        for (int f = 0; f < FEATURE_DIM; f++) {
          interpolated_features[FEATURE_DIM*q_idx + f] += uv_curr * V_continuous_feat[FEATURE_DIM*v_idx + f];
        }
      } else {
        // Discontinuous vertex
        int cw_idx = tri_idx*6 + k*2 + 0;
        int ccw_idx = tri_idx*6 + k*2 + 1;

        // Find the vector for the directed edge corresponding to the CW and CCW features
        float2 disc_cw_vec  = V[T_adj_disc_V[cw_idx]] - v;
        float2 disc_ccw_vec = V[T_adj_disc_V[ccw_idx]] - v;

        // This angle is correct for BOTH CW AND CCW, not a bug.
        float disc_cw_angle = get_angle_ccw(disc_cw_vec, q - v);
        float disc_ccw_angle = get_angle_ccw(q - v, disc_ccw_vec);

        float w_cw = disc_ccw_angle / (disc_ccw_angle + disc_cw_angle + EPS_DIV);
        float w_ccw = 1.0 - w_cw;

        int offset_cw  = T_adj_disc_feat_idx[cw_idx];
        int offset_ccw = T_adj_disc_feat_idx[ccw_idx];
        for (int f = 0; f < FEATURE_DIM; f++) {
          interpolated_features[FEATURE_DIM*q_idx + f] += uv_curr * (
            w_cw  * V_discontinuous_feat[2*FEATURE_DIM*offset_cw + f] +
            w_ccw * V_discontinuous_feat[2*FEATURE_DIM*offset_ccw + FEATURE_DIM + f]
          );
        }
      }
    }
  }
}