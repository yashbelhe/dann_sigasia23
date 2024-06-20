static const float BEPS = 1e-5;
// Note: Consider using double precision for positions to get better point in triangle tests.
// The parameter below is quite sensitive, make sure it is set correctly for high zoom levels
static const float TRI_TEST_EPS = 1e-10;
// static const float TRI_TEST_EPS = 1e-8;
static const float CUBIC_FACTOR = 1000.0;
// static const float TRI_TEST_EPS = 1e-5;

#define M_PI 3.14159265358979323846
// #define ATAN_EPS 1e-3
#define ATAN_EPS 1e-7
// #define ATAN_EPS 1e-6
// #define ATAN_EPS 1e-4

double three_C_k(int k) {
  if (k == 0 || k == 3) return 1.0;
  else if (k == 1 || k == 2) return 3.0;
  else return 0.0 / 0.0;
}

float2 l2b(float2 v0, float2 v1, float2 v2, float2 v3, int idx) {
  if (idx == 1) {
    return -5.0 / 6.0 * v0 + 3.0 * v1 - 3.0 / 2.0 * v2 + 1.0 / 3.0 * v3;
  } else if (idx == 2) {
    return -5.0 / 6.0 * v3 + 3.0 * v2 - 3.0 / 2.0 * v1 + 1.0 / 3.0 * v0;
  }
  return float2(-1.0);
}

double l_ij(float2 p, float2 p_i, float2 p_j, int i, int j) {
    double x = p.x, y = p.y, x_i = p_i.x, y_i = p_i.y, x_j = p_j.x, y_j = p_j.y;
  return three_C_k(i) * three_C_k(j) * (x * (y_i - y_j) - y * (x_i - x_j) + x_i * y_j - x_j * y_i);
}

double implicit_cubic(float2 p, float2 p0, float2 p1, float2 p2, float2 p3) {
    double l_21 = l_ij(p, p2, p1, 2, 1);
    double l_10 = l_ij(p, p1, p0, 1, 0);
    double l_20 = l_ij(p, p2, p0, 2, 0);
    double l_30 = l_ij(p, p3, p0, 3, 0);
    double l_31 = l_ij(p, p3, p1, 3, 1);
    double l_32 = l_ij(p, p3, p2, 3, 2);
    double c = l_30 + l_21;
  return l_32*(c*l_10 - l_20*l_20) - l_31*(l_31*l_10 - l_20*l_30) + l_30*(l_31*l_20 - c*l_30);
}


// domain_boundary_idx
int dbi(float2 x) {
  if (x.x < BEPS) return 0;
  else if (x.y < BEPS) return 1;
  else if (1.0 - x.x < BEPS) return 2;
  else if (1.0 - x.y < BEPS) return 4;
  return -1;
}
// is on domain boundary
bool idb(float2 x) {
  if (x.x < BEPS) return true;
  else if (x.y < BEPS) return true;
  else if (1.0 - x.x < BEPS) return true;
  else if (1.0 - x.y < BEPS) return true;
  return false;
}

float get_angle_ccw(float2 v1, float2 v2) {
  // double2 _v1 = double2(v1);
  // double2 _v2 = double2(v2);
  // double val = atan2(_v1.x*_v2.y - _v1.y*_v2.x, dot(_v1, _v2));
  float val = atan2(v1.y*v2.x - v1.x*v2.y, dot(v1, v2));
  // float val = atan2(v1.x*v2.y - v1.y*v2.x, dot(v1, v2));
  // float val = atan2(v1.x*v2.y - v1.y*v2.x, dot(v1, v2));
  if (val >= -ATAN_EPS) {
  // if (val >= 0.0) {
    return val;
  }
  return 2.0 * M_PI + val;
}


struct BBox {
    float2 min_xy;
    float2 max_xy;
    int2 min_xy_idx;
    int2 max_xy_idx;
}

BBox get_bounding_box(float2 v0, float2 v1, float2 v2, int kX, int kY) {
    float2 min_xy = float2(min(min(v0.x, v1.x), v2.x), min(min(v0.y, v1.y), v2.y));
    float2 max_xy = float2(max(max(v0.x, v1.x), v2.x), max(max(v0.y, v1.y), v2.y));
    int2 min_xy_idx = int2(int(min_xy.x * kX), int(min_xy.y * kY));
    int2 max_xy_idx = int2(int(max_xy.x * kX), int(max_xy.y * kY));
    // int2 max_xy_idx = int2(int(max_xy.x * kX) + 1, int(max_xy.y * kY) + 1);
    BBox ret = { min_xy, max_xy, min_xy_idx, max_xy_idx };
    return ret;
}

bool is_inside_triangle(float2 q, float2 v1, float2 v2, float2 v3) {
    // 2D triangle test (return 1.0 if the triangle contains the point (x,y), 0.0 otherwise).
    float2 e0 = v2 - v1;
    float2 e1 = v3 - v2;
    float2 e2 = v1 - v3;

    float2 k0 = float2(-e0.y, e0.x);
    float2 k1 = float2(-e1.y, e1.x);
    float2 k2 = float2(-e2.y, e2.x);

    float d0 = dot(k0, q - v1);
    float d1 = dot(k1, q - v2);
    float d2 = dot(k2, q - v3);

    // Check the three half-plane values.
    // if (d0 >= -TRI_TEST_EPS && d1 >= -TRI_TEST_EPS && d2 >= -TRI_TEST_EPS)
    if (d0 <= TRI_TEST_EPS && d1 <= TRI_TEST_EPS && d2 <= TRI_TEST_EPS)
        return true;
    else
        return false;
}

// Parallelogram law
float area(float2 v1, float2 v2) {
    return v1[0] * v2[1] - v1[1] * v2[0];
}

float3 clip(float3 x, float l, float u) {
    return float3(max(min(x.x, u), l), max(min(x.y, u), l), max(min(x.z, u), l));
}

float3 sum_one(float3 x) {
    return x / (x.x + x.y + x.z);
}

// Taken from: https://github.com/postmalloc/barycuda/blob/master/src/bary.cu
float3 get_barycentrics(float2 q, float2 a, float2 b, float2 c) {
    float2 aq = q - a;
    float2 bq = q - b;
    float2 cq = q - c;

    float2 ac = c - a;
    float2 ab = b - a;
    float2 ca = a - c;
    float2 bc = c - b;

    float nor  = area(ab, ac);
    float nor0 = area(bc, bq);
    float nor1 = area(ca, cq);
    float nor2 = area(ab, aq);

    // return float3(nor0, nor1, nor2) / nor;
    return sum_one(clip(float3(nor0, nor1, nor2) / nor, 0.0, 1.0));
}