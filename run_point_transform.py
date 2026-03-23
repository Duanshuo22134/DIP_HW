import cv2
import numpy as np
import gradio as gr

# Global variables for storing source and target control points
points_src = []
points_dst = []
image = None

# Reset control points when a new image is uploaded
def upload_image(img):
    global image, points_src, points_dst
    points_src.clear()
    points_dst.clear()
    image = img
    return img

# Record clicked points and visualize them on the image
def record_points(evt: gr.SelectData):
    global points_src, points_dst, image
    x, y = evt.index[0], evt.index[1]

    # Alternate clicks between source and target points
    if len(points_src) == len(points_dst):
        points_src.append([x, y])
    else:
        points_dst.append([x, y])

    # Draw points (blue: source, red: target) and arrows on the image
    marked_image = image.copy()
    for pt in points_src:
        cv2.circle(marked_image, tuple(pt), 1, (255, 0, 0), -1)  # Blue for source
    for pt in points_dst:
        cv2.circle(marked_image, tuple(pt), 1, (0, 0, 255), -1)  # Red for target

    # Draw arrows from source to target points
    for i in range(min(len(points_src), len(points_dst))):
        cv2.arrowedLine(marked_image, tuple(points_src[i]), tuple(points_dst[i]), (0, 255, 0), 1)

    return marked_image

# Point-guided image deformation
def point_guided_deformation(image, source_pts, target_pts, alpha=1.0, eps=1e-8):
    """
    MLS (Moving Least Squares) 变形算法实现，基于控制点对图像进行局部变形。
    
    Parameters
    ----------
    image      : np.ndarray, 输入图像 (H, W) 或 (H, W, C)
    source_pts : np.ndarray, 形状 (n, 2)，源控制点坐标 (x, y)
    target_pts : np.ndarray, 形状 (n, 2)，目标控制点坐标 (x, y)
    alpha      : float, 权重衰减指数
    eps        : float, 防止分母为0

    Returns
    -------
    warped_image : np.ndarray, 变形后的图像
    """
    warped_image = np.zeros_like(image)
    h, w = image.shape[:2]

    # 若无控制点，直接返回原图
    if len(source_pts) == 0 or len(target_pts) == 0:
        return image.copy()

    # source_pts: 变形前点 p，target_pts: 变形后点 q
    p = source_pts.astype(np.float64)   # shape (n, 2)
    q = target_pts.astype(np.float64)   # shape (n, 2)
    n = p.shape[0]
    
    # 处理控制点数量不匹配的情况
    num_pts = min(len(p), len(q))
    p = p[:num_pts]
    q = q[:num_pts]

    # -------------------------------------------------------
    # 构造输出图像所有像素坐标网格
    # -------------------------------------------------------
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))

    # v: 所有像素点，shape (H*W, 2)，每行是一个 (x, y)，表示像素点坐标
    v = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1).astype(np.float64)

    # -------------------------------------------------------
    # Step 1: 计算每个像素 v 到每个控制点 p_i 的权重 w_i
    # w_i(v) = 1 / ||v - p_i||^(2*alpha)
    # -------------------------------------------------------
    diff = v[:, np.newaxis, :] - p[np.newaxis, :, :]   
    dist2 = np.sum(diff ** 2, axis=2)                   
    weights = 1.0 / (dist2 ** alpha + eps)              

    # -------------------------------------------------------
    # Step 2: 计算加权质心 p* 和 q*
    # p* = Σ(w_i * p_i) / Σw_i      — 源点质心
    # q* = Σ(w_i * q_i) / Σw_i      — 目标点质心
    # -------------------------------------------------------
    w_sum = np.sum(weights, axis=1, keepdims=True)          # (M, 1)
    p_star = np.sum(weights[:, :, np.newaxis] * p[np.newaxis, :, :], axis=1) / w_sum  # (M, 2)
    q_star = np.sum(weights[:, :, np.newaxis] * q[np.newaxis, :, :], axis=1) / w_sum  # (M, 2)

    # -------------------------------------------------------
    # Step 3: 去质心后的点
    # p_hat_i = p_i - p*   q_hat_i = q_i - q*
    # -------------------------------------------------------
    p_hat = p[np.newaxis, :, :] - p_star[:, np.newaxis, :]  # (M, n, 2)
    q_hat = q[np.newaxis, :, :] - q_star[:, np.newaxis, :]  # (M, n, 2)

    # -------------------------------------------------------
    # Step 4: 计算最优仿射变换矩阵 M（闭式解）
    # M = (Σ w_i * p̂_i^T * p̂_i)^{-1} * (Σ w_j * p̂_j^T * q̂_j)
    # -------------------------------------------------------
    W = np.einsum('mi,mij,mik->jk', weights, p_hat, p_hat)  # (2, 2)
    J = np.einsum('mi,mij,mik->jk', weights, p_hat, q_hat)  # (2, 2)
    
    try:
        M = np.linalg.solve(W, J)
    except np.linalg.LinAlgError:
        # 如果W奇异，使用伪逆
        M = np.linalg.pinv(W) @ J
    
    # -------------------------------------------------------
    # Step 5: 反向映射坐标
    # src = M @ (v - p*) + q*
    # -------------------------------------------------------
    v_hat = v - p_star  # (M, 2)
    src = (v_hat @ M.T) + q_star  # (M, 2)
    
    src_x = src[:, 0].reshape(h, w)
    src_y = src[:, 1].reshape(h, w)

    # -------------------------------------------------------
    # Step 6: 双线性插值采样
    # 平滑边界，取坐标周围四个像素点的均值
    # 取整数坐标（左上角）
    x0 = np.floor(src_x).astype(np.int32)
    y0 = np.floor(src_y).astype(np.int32)
    x1 = x0 + 1
    y1 = y0 + 1

    # 插值权重（小数部分）
    wx = (src_x - x0).astype(np.float32)   # (H, W)
    wy = (src_y - y0).astype(np.float32)   # (H, W)

    # 边界裁剪，防止越界
    x0c = np.clip(x0, 0, w - 1)
    x1c = np.clip(x1, 0, w - 1)
    y0c = np.clip(y0, 0, h - 1)
    y1c = np.clip(y1, 0, h - 1)

    # 扩展维度以支持多通道
    wx = wx[:, :, np.newaxis]   # (H, W, 1)
    wy = wy[:, :, np.newaxis]   # (H, W, 1)

    # 四邻域采样
    I00 = image[y0c, x0c].astype(np.float32)   # 左上
    I10 = image[y0c, x1c].astype(np.float32)   # 右上
    I01 = image[y1c, x0c].astype(np.float32)   # 左下
    I11 = image[y1c, x1c].astype(np.float32)   # 右下

    # 双线性插值
    warped_image = (
        (1 - wx) * (1 - wy) * I00 +
        wx       * (1 - wy) * I10 +
        (1 - wx) * wy       * I01 +
        wx       * wy       * I11
    ).astype(np.uint8)

    return warped_image



def run_warping():
    global points_src, points_dst, image

    warped_image = point_guided_deformation(image, np.array(points_dst), np.array(points_src))

    return warped_image

# Clear all selected points
def clear_points():
    global points_src, points_dst
    points_src.clear()
    points_dst.clear()
    return image

# Build Gradio interface
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Upload Image", interactive=True, width=800)
            point_select = gr.Image(label="Click to Select Source and Target Points", interactive=True, width=800)

        with gr.Column():
            result_image = gr.Image(label="Warped Result", width=800)

    run_button = gr.Button("Run Warping")
    clear_button = gr.Button("Clear Points")

    input_image.upload(upload_image, input_image, point_select)
    point_select.select(record_points, None, point_select)
    run_button.click(run_warping, None, result_image)
    clear_button.click(clear_points, None, point_select)

demo.launch()
