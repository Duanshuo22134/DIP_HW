## Implementation of Image Geometric Transformation

This repository is Chang Huayijun's implementation of Assignment_02 of DIP - Poisson Editing.


## Requirements

To install requirements:

```setup
python -m pip install -r requirements.txt
```

## Running

To run Poisson Editing, run:

```basic
python run_blending_gradio.py
```


## Results

### Image Fusion - Poisson Editing
<img src="poisson_editing.gif" alt="Image fusion demo" width="800">

The demo shows image fusion using Poisson Editing.


## Key Technical Details

### Poisson Blending Pipeline

This project implements an interactive Poisson image blending system. The user first draws a polygon on the foreground image, then shifts the selected region onto the background with offsets $(dx, dy)$, and finally optimizes the pasted region so that its gradient field matches the source object while remaining visually consistent with the target image.

1. Polygon-based region selection

The object region is defined by user clicks on the foreground image. After the polygon is closed, the vertices are rasterized into a binary mask:

$$
\Omega_f(x, y) =
\begin{cases}
1, & (x, y)\ \text{inside the polygon}\\
0, & \text{otherwise}
\end{cases}
$$

In the implementation, OpenCV `fillPoly` is used to convert polygon vertices into the foreground mask.

2. Region translation to the target image

If the polygon is moved by $(dx, dy)$, the same vertices are translated to the background coordinate system:

$$
\mathbf{p}_i^{(bg)} = \mathbf{p}_i^{(fg)} + (dx, dy)
$$

This produces a target mask $\Omega_b$. The source image and its mask are then copied into the background canvas at the translated location, which guarantees that the source tensor, target tensor, and optimization variable share the same spatial size.

3. Gradient-domain objective

Instead of directly matching pixel intensities, Poisson editing preserves the source object's local structure by matching Laplacians inside the pasted region. The discrete Laplacian kernel used in the code is:

$$
K =
\begin{bmatrix}
0 & 1 & 0\\
1 & -4 & 1\\
0 & 1 & 0
\end{bmatrix}
$$

For each RGB channel, the optimization minimizes the difference between the Laplacian of the shifted foreground image $S$ and the Laplacian of the blended result $B$:

$$
\mathcal{L} =
\frac{1}{|\Omega|}
\sum_{(x,y)\in\Omega}
\left|
\Delta S(x,y) - \Delta B(x,y)
\right|
$$

where $\Omega = \Omega_f \cap \Omega_b$ is the valid pasted region. In code, the Laplacian is computed by grouped `conv2d`, so each color channel is filtered independently.

4. Boundary handling

Only pixels inside the pasted mask are optimized. Pixels outside the mask are fixed to the original background:

$$
B'(x,y) = (1-\Omega_b(x,y))\,T(x,y) + \Omega_b(x,y)\,B(x,y)
$$

This masking strategy keeps the background unchanged and restricts optimization to the selected region, which is the core requirement of Poisson image editing.

5. Optimization strategy

The blended image is initialized from the background image, with a small amount of source color injected into the masked region for a better starting point. The final image is obtained by directly optimizing pixel values with Adam:

$$
B^{*} = \arg\min_B \mathcal{L}(B)
$$

The current implementation uses 5000 optimization steps and decays the learning rate after two thirds of the iterations. CUDA is used when available; otherwise the same procedure runs on CPU, although it is much slower.

6. Practical implementation notes

- The foreground is shifted onto a full background-sized canvas before optimization, avoiding shape mismatch between source and target tensors.
- The foreground mask is clipped by the translated background mask so that invalid out-of-bound areas are excluded.
- The result is clamped to $[0, 1]$ after optimization and converted back to an 8-bit RGB image for display in Gradio.

## Acknowledgement

> Thanks for the idea of gradient-domain image compositing introduced by [Paper: Poisson Image Editing](https://www.cs.jhu.edu/~misha/Fall07/Papers/Perez03.pdf)

**Reference**: Patrick Perez, Michel Gangnet, Andrew Blake. *Poisson Image Editing*. ACM SIGGRAPH, 2003.

