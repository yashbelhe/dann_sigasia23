import torch
import pydiffvg
import math
import numpy as np
from svgpathtools import svg2paths, Line, Arc
from PIL import Image
import os

import time

def get_locs_zoom(x, y, width):
  hw = width/2
  return (x - hw, x + hw, y - hw, y+ hw)

def get_stratified_random(n_sqrt, device=torch.device("cuda")):
  x = torch.linspace(0, 1, n_sqrt, device=device)
  y = torch.linspace(0, 1, n_sqrt, device=device)
  xx, yy = torch.meshgrid(x, y)
  xx, yy = xx.reshape(-1), yy.reshape(-1)
  xx = xx + torch.rand_like(xx) / n_sqrt
  yy = yy + torch.rand_like(yy) / n_sqrt
  return torch.stack([xx, yy], axis=-1)

def get_stratified_in_triangles(model, device=torch.device("cuda")):
  triangles = model.T
  vertices = model.V
  N = triangles.shape[0]
  u1 = torch.rand(N, device=device).unsqueeze(1)
  u2 = torch.rand(N, device=device).unsqueeze(1)
  sqrt_u1 = torch.sqrt(u1)
  tri_idxs = torch.arange(N, device=device)
  ABC = vertices[triangles[tri_idxs].long()]
  ABC = torch.tensor(ABC, device=device)
  res = ABC[:,0]*(1.0 - sqrt_u1) + ABC[:,1]*(1.0 - u2)*sqrt_u1 + ABC[:,2]*u2*sqrt_u1
  return res

class BaseSampler:
  def __call__(self):
    raise NotImplementedError

  def plot_locs(self):
    return [
      (0.0, 1.0, 0.0, 1.0),
      *[get_locs_zoom(0.4182068, 0.7723169, zoom) for zoom in [0.5, 0.2, 0.1, 0.01]]
    ]

################ Vector Graphics #############

class VGSampler(BaseSampler):
  def __init__(self, fname):
    pydiffvg.set_device(torch.device('cuda:0'))
    svg_fname = fname.split(".")[0] + '.svg'
    print(f"Loaded SVG file: {svg_fname}")
    self.w, self.h, self.sh, self.sh_grp = pydiffvg.svg_to_scene(svg_fname)
  
  def __call__(self, Q):
    x = Q[:,0]*self.w
    # x = (1.0 - Q[:,0])*self.w
    y = Q[:,1]*self.h
    xy = torch.stack([y,x], axis=-1)
    scene_args = pydiffvg.RenderFunction.serialize_scene(self.w, self.h, self.sh, self.sh_grp, eval_positions=xy)
    samples = pydiffvg.RenderFunction.apply(self.w, self.h, 0, 0, 0, None, *scene_args)
    return samples[:,:3]

############# Rendering ##############

class RenderingSampler(BaseSampler):
  def __init__(self, fname):
    print("Start: Loading Rendering Sampler")
    self.S = 50000
    self.img = torch.load('data/rendering/flowerpot/img.pt').cuda()
    print("End: Loading Rendering Sampler")
  
  def __call__(self, Q):
    _x, _y = Q[:,0], Q[:,1]
    _x_idx = torch.clamp((_x * self.S).type(torch.long).cpu(), 0, self.S - 1)
    _y_idx = torch.clamp((_y * self.S).type(torch.long).cpu(), 0, self.S - 1)
    res = self.img[_x_idx, _y_idx]
    res = res.type(torch.float32)/255.0
    return res

######### Walk on Spheres #########
class WoSScene:
    lines = None
    is_line = None
    left_img = None
    right_img = None


class WoSSampler(BaseSampler):
  def __init__(self, fname, device=torch.device('cuda')):
    svg_fname = fname.split(".")[0] + '.svg'
    base_dir = fname.split("/")[:-1]
    paths, attributes = svg2paths(svg_fname)
    segments = []
    segment_line = []
    self.max_walk_length = 20
    self.eps = 1e-3
    self.spp = 10
    # self.spp = 100
    W = 1000
    for path in paths:
      for subpath in path:
        if type(subpath) == Line:
          segments.append(
              (torch.tensor([subpath.start.real/W, subpath.start.imag/W], device=device, dtype=torch.float),
              torch.tensor([subpath.end.real/W, subpath.end.imag/W], device=device, dtype=torch.float)))
          segment_line.append(True)
        elif type(subpath) == Arc:
          segments.append((torch.tensor([subpath.center.real/W, subpath.center.imag/W], device=device, dtype=torch.float),
                          torch.tensor(subpath.radius.real/W, device=device, dtype=torch.float)))
          segment_line.append(False)
          break
        else:
           print(type(subpath))
           continue
          #  assert False, "Walk on Spheres Loader, unsupported segment type"
    self.wos_scene = WoSScene()
    self.wos_scene.lines = segments
    self.wos_scene.is_line = segment_line
    self.wos_scene.right_img = torch.tensor(np.asarray(Image.open(os.path.join(*base_dir, 'left.png')))
                        [:, :, :3]/255.0, device=device,dtype=torch.float32).transpose(0, 1)
    self.wos_scene.left_img = torch.tensor(np.asarray(Image.open(os.path.join(*base_dir, 'right.png')))
                    [:, :, :3]/255.0, device=device,  dtype=torch.float32).transpose(0, 1)
    
  def __call__(self, Q):
    XY = Q[:,[1,0]]
    return walk_on_spheres(XY, self.spp, self.max_walk_length, self.eps, self.wos_scene)

def dot(x, y):
  return torch.sum(x * y, axis=1)

def length(x):
  return torch.sqrt(torch.sum(torch.square(x), axis=1))


def line_segment(p, a, b):
  a = a.unsqueeze(0)
  b = b.unsqueeze(0)
  ba = b - a
  pa = p - a
  h = torch.clamp(dot(pa, ba) / (dot(ba, ba) + 1e-8), 0., 1.).unsqueeze(1)
  return length(pa - h * ba)

def line_implicit(p, a, b):
  x1, y1 = a[0], a[1]
  x2, y2 = b[0], b[1]

  return (y2-y1)*p[:, 0] - (x2-x1)*p[:, 1] + x2*y1 - x1*y2


def sdCircle(p, c, r):
  c = c.unsqueeze(0)
  return length(p - c) - r


def walk_on_spheres(XY, spp, max_walk_length, eps, scene_obj, device=torch.device('cuda')):
  def scene(p, scene_obj):
      d = 1e-2
      lines = scene_obj.lines
      is_line = scene_obj.is_line
      img_w = scene_obj.left_img.shape[0]
      img_h = scene_obj.left_img.shape[1]

      def color(p, side):
        px = p[:, 0]
        py = p[:, 1]
        x = torch.clip(torch.round(px * img_w), 0, img_w - 1).type(torch.long)
        y = torch.clip(torch.round(py * img_h), 0, img_h - 1).type(torch.long)

        return torch.where(side.unsqueeze(1), scene_obj.left_img[x, y], scene_obj.right_img[x, y])

      min_dist = 10000 + torch.zeros(p.shape[0], device=device)
      min_colors = torch.zeros((p.shape[0], 3), device=device)
      for line_idx, line in enumerate(lines):
        if is_line[line_idx]:
          curr_dist = line_segment(p, *line)
          assert not torch.any(torch.isnan(curr_dist))
          side_line = line_implicit(p, *line) <= 0
        else:
          sd = sdCircle(p, *line)
          curr_dist = torch.abs(sd)
          assert not torch.any(torch.isnan(sd))
          side_line = sd <= 0
        curr_color = color(p, side_line)
        # curr_closest_point = closest_point_on_line(p, *line)
        min_colors = torch.where(
            (curr_dist < min_dist).unsqueeze(1), curr_color, min_colors)
        min_dist = torch.minimum(min_dist, curr_dist)
      return min_dist, min_colors
  
  p = torch.repeat_interleave(XY, spp, dim=0)

  N = p.shape[0]

  final_color = torch.zeros((N, 3), dtype=torch.float32, device=device)
  temp_color = torch.zeros((N, 3), dtype=torch.float32, device=device)
  temp_dist = torch.zeros(N, dtype=torch.float32, device=device)
  temp_hitb = torch.zeros(N, dtype=torch.bool, device=device)
  active = torch.ones(N, dtype=bool, device=device)
  for walk_length in range(max_walk_length):
    if active.sum() == 0:
      break
    dist, color = scene(p[active], scene_obj)

    hit_b = dist < eps
    temp_color.zero_()
    temp_dist.zero_()
    temp_hitb.zero_()
    temp_color[active] = color
    temp_dist[active] = dist
    temp_hitb[active] = hit_b
    final_color = torch.where(torch.logical_and(
        active, temp_hitb).unsqueeze(1), temp_color, final_color)
    active[temp_hitb] = False

    theta = 2 * torch.pi * torch.rand(N, device=device)
    delta_p = temp_dist.unsqueeze(1) * torch.stack([torch.cos(theta), torch.sin(theta)], axis=1)
    p = p + delta_p
  return torch.mean(final_color.reshape(XY.shape[0], spp, 3), axis=1)