# Implementation of code for Discontinuity-Aware 2D Neural Fields (SIGGRAPH Asia 2023, Transactions on Graphics) [website](https://yashbelhe.github.io/danf/index.html), [paper](https://yashbelhe.github.io/danf/DiscontinuityAwareNeuralFields_SigAsia2023.pdf), [slides PDF](https://yashbelhe.github.io/danf/index.html).
### Authors: Yash Belhe, Michael Gharbi, Matt Fisher, Iliyan Georgiev, Ravi Ramamoorthi, Tzu-Mao Li

This code is a re-implementation of the paper using SLANG and PyTorch. As such, the results may not exactly match the original implementation.

Using this code, you can (qualitatively) reproduce a few examples from the paper:
1. Flowerpot scene (Fig. 9) -- Rendering
2. Circles scene (Fig. 10) -- Walk on Spheres
3. Shapes scene (Fig. 2) -- Vector Graphics

## Setup
Install pytorch and [diffvg](https://github.com/BachiLi/diffvg) (with python bindings).

```pip install scikit-image numpy matplotlib slangpy svgpathtools pillow```

## Data
Download the data from here and place it in the root directory. 

## Run
To run the circles scene: `python train.py circles` and similarly for `shapes, flowerpot`.

### Notable missing components:
1. Mesh compression using draco.
2. Data preparation for custom scenes, a> modified version of TriWild and b> edge extraction for rendering scenes.

```
@article{Belhe:2023:DiscontinuityAwareNeuralFields,
author = {Yash Belhe and Micha\"{e}l Gharbi and Matthew Fisher and Iliyan Georgiev and Ravi Ramamoorthi and Tzu-Mao Li},
title = {Discontinuity-aware 2D neural fields},
journal = {ACM Transactions on Graphics (Proceedings of SIGGRAPH Asia)},
year = {2023},
volume = {41},
number = {6},
doi = {10.1145/3550454.3555484}
}
```