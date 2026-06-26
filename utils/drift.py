import numpy as np
from scipy.ndimage import shift, uniform_filter1d


def adapt_imcrop(im_uncroped, shift_vector, crop_method='square'):
    """
    根据位移向量自动计算裁剪区域，剔除边缘无效像素

    参数:
    im_uncroped: 待裁剪的平均 PSF 图像 (H, W)
    shift_vector: 所有帧的位移记录 (Frames, 2), 顺序为 [dx, dy]
    """
    # 1. 计算最大平移量 (对应 MATLAB 的逻辑)
    max_shift = np.max(np.abs(shift_vector))

    if crop_method == 'max':
        # 对应 MATLAB case 'max'
        margin = int(np.ceil(max_shift)) + 1
    else:
        # 默认 'square' 逻辑
        margin = int(np.ceil(max_shift))

    h, w = im_uncroped.shape

    # 2. 定义裁剪边界 [y_start : y_end, x_start : x_end]
    # 注意：margin 是从四周往里缩进的距离
    y_start, y_end = margin, h - margin
    x_start, x_end = margin, w - margin

    if y_start >= y_end or x_start >= x_end:
        raise ValueError("最大位移过大，导致可用图像区域消失，请检查序列输入！")

    im_croped = im_uncroped[y_start:y_end, x_start:x_end]

    # 返回裁剪后的图像以及裁剪参数（供以后坐标对齐用）
    crop_info = {'margin': margin, 'original_shape': (h, w)}
    return im_croped, crop_info


def drift_correction(data_fit, pixel_size, fitter, window_size=3, max_shift_pix=10):
    """
    通用漂移修正：复刻 MATLAB 逻辑 (先平滑图像序列，再寻心拟合)
    支持 PSF (H, W, Frames) 和 LPM (H, W, Channels, Frames)
    """
    ndim = data_fit.ndim

    # ---------------------------------------------------------
    # 1. 维度检测与数据格式化
    # ---------------------------------------------------------
    if ndim == 3:
        h, w, frames = data_fit.shape
        n_channels = 1
        data_fit = data_fit[:, :, np.newaxis, :]  # 统一转为 4D 处理
        print(f"正在处理 PSF 数据 (帧数: {frames})...")
    elif ndim == 4:
        h, w, n_channels, frames = data_fit.shape
        data_fit = data_fit
        max_shift_pix = 5
        window_size = 30
        print(f"正在处理 LPM 数据 (通道: {n_channels}, 帧数: {frames})...")
    else:
        raise ValueError(f"不支持的维度: {ndim}")

    # ---------------------------------------------------------
    # 2. 核心：先平滑图像数据
    # ---------------------------------------------------------
    print(f"正在执行时间轴平滑 (window={window_size})...")
    # 只取通道 0 用于寻心，并在时间轴 (axis=2) 上平滑
    # 注意：data_fit[:,:,0,:] 的维度是 (H, W, Frames)
    data_to_fit = uniform_filter1d(data_fit[:, :, 0, :].astype(float), size=window_size, axis=2)

    # ---------------------------------------------------------
    # 3. 寻心拟合：使用平滑后的图像
    # ---------------------------------------------------------
    if ndim == 3:
        print("执行 PSF (Poly22) 寻心...")
        centers = fitter.fit22(data_to_fit, mask_radius=120)
    else:
        print("执行 LPM (Poly44) 寻心...")
        # 注意：fitter.fit44 内部已修改为取输入的通道 0 或直接处理 3D
        centers = fitter.fit44(data_to_fit, mask_radius=120)

    # ---------------------------------------------------------
    # 4. 计算位移向量与有效帧判定
    # ---------------------------------------------------------
    # 物理位移转像素位移 (取负、除以像素、四舍五入)
    shift_vectors = np.round(-centers / pixel_size).astype(int)

    # 判定有效帧 (复刻 MATLAB: shiftMax <= maxShiftPix)
    shift_max = np.max(np.abs(shift_vectors), axis=1)
    valid_mask = shift_max <= max_shift_pix
    invalid_count = np.sum(~valid_mask)

    if invalid_count > 0:
        print(f"剔除无效帧: {invalid_count} / {frames}")

    # 仅保留有效帧的位移记录
    shift_vectors_valid = shift_vectors[valid_mask]
    if len(shift_vectors_valid) == 0:
        raise ValueError("所有帧均超过最大位移阈值，请检查数据！")

    # ---------------------------------------------------------
    # 5. 统一执行平移 (作用于原始 data_fit)
    # ---------------------------------------------------------
    # data_shifted 只存储有效帧
    data_shifted = np.zeros((h, w, n_channels, np.sum(valid_mask)))

    valid_indices = np.where(valid_mask)[0]
    for c in range(n_channels):
        for jj, j in enumerate(valid_indices):
            dy, dx = int(shift_vectors[j, 1]), int(shift_vectors[j, 0])
            # order=1 保持插值平滑性
            data_shifted[:, :, c, jj] = shift(data_fit[:, :, c, j], shift=[dy, dx], order=1)

    # ---------------------------------------------------------
    # 6. 自适应裁剪 (调用 adapt_imcrop)
    # ---------------------------------------------------------
    # 使用平移后第一帧参考通道的图像来确定边界 (对应 MATLAB: dataSmoothed(:,:,1))
    ref_img_for_crop = data_shifted[:, :, 0, 0]

    try:
        _, crop_info = adapt_imcrop(ref_img_for_crop, shift_vectors_valid, crop_method='square')
        m = int(crop_info['margin'])

        if m > 0:
            data_cropped = data_shifted[m:-m, m:-m, :, :]
        else:
            data_cropped = data_shifted.copy()

        print(f"自适应裁剪完成，尺寸: {data_cropped.shape[0]}x{data_cropped.shape[1]}")
    except ValueError as e:
        print(f"裁剪失败，保留原尺寸: {e}")
        data_cropped = data_shifted

    # ---------------------------------------------------------
    # 7. 结果还原
    # ---------------------------------------------------------
    # 计算最终平均图
    psf_avg_final = np.mean(data_cropped, axis=-1)

    if ndim == 3:
        # psf_avg_final: (H_new, W_new)
        # data_cropped: (H_new, W_new, Frames_valid)
        return psf_avg_final.squeeze(), data_to_fit.squeeze(), data_cropped.squeeze(), centers, shift_vectors
    else:
        # psf_avg_final: (H_new, W_new, 4)
        # data_cropped: (H_new, W_new, 4, Frames_valid)
        return psf_avg_final, data_to_fit, data_cropped, centers, shift_vectors
