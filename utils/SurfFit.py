import numpy as np
from scipy.optimize import minimize
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

class SurfFit:
    def __init__(self, xpixels, ypixels, pixelsize):
        """
        构造函数：建立与 MATLAB 相同的中心对齐坐标系
        """
        x_end = pixelsize * (xpixels - 1) / 2
        y_end = pixelsize * (ypixels - 1) / 2

        self.x = np.linspace(-x_end, x_end, xpixels)
        self.y = np.linspace(-y_end, y_end, ypixels)
        self.xx, self.yy = np.meshgrid(self.x, self.y)

    def _fit_logic(self, psf, degree, mask_radius):
        """
        极速优化版：批量拟合多帧数据
        """
        ndim = psf.ndim
        if ndim == 4:
            # psf 维度: (H, W, Channels, Frames)
            # 取第一个通道，将其降维至 (H, W, Frames)
            psf_to_fit = psf[:, :, 0, :]
        elif ndim == 3:
            # psf 维度: (H, W, Frames)
            psf_to_fit = psf
        elif ndim == 2:
            psf_to_fit = psf[:, :, np.newaxis]
        else:
            raise ValueError(f"SurfFit 不支持的维度: {ndim}")

        h, w, frames = psf_to_fit.shape

        centers = np.zeros((frames, 2))

        x_flat = self.xx.ravel()
        y_flat = self.yy.ravel()

        # 1. 生成 Mask
        if mask_radius is not None:
            mask = (x_flat ** 2 + y_flat ** 2) <= mask_radius ** 2
        else:
            mask = np.ones_like(x_flat, dtype=bool)

        # 2. 准备多项式特征 (只做一次)
        x_fit = x_flat[mask]
        y_fit = y_flat[mask]
        poly = PolynomialFeatures(degree=degree)
        features = poly.fit_transform(np.column_stack((x_fit, y_fit)))

        # 3. 【关键优化】：将所有帧的 z 轴数据一次性提取出来
        # 先将 psf 重塑为 (h*w, frames)，然后应用 mask 筛选像素
        # 得到的 all_z_fit 形状为 (选中的像素数, frames)
        all_z_fit = psf_to_fit.reshape(-1, frames)[mask, :]

        # 4. 【核心加速】：一次性拟合所有帧
        # LinearRegression 的 fit 可以接受多列输出 y
        model = LinearRegression(fit_intercept=False)
        model.fit(features, all_z_fit)

        # 拿到所有帧的系数矩阵：coeffs_all 的形状是 (frames, 15)
        # 注意：多输出拟合时，coef_ 的形状通常是 (n_outputs, n_features)
        coeffs_all = model.coef_

        # 5. 寻心环节 (寻优器目前还是需要逐帧跑，但拟合已经瞬间完成了)
        for i in range(frames):
            current_coeffs = coeffs_all[i]

            # 这里我们可以直接使用你之前的 poly44 逻辑，或者继续用 predict
            # 但为了极速，我们可以定义一个局部函数直接计算
            def f(coords):
                # 将输入的 [x, y] 转换为对应的多项式特征向量
                c_feat = poly.transform(np.array(coords).reshape(1, -1))
                val = np.dot(c_feat, current_coeffs)
                return val[0]

            res = minimize(
                f,
                x0=[0, 0],
                bounds=[(self.x.min(), self.x.max()), (self.y.min(), self.y.max())],
                method='L-BFGS-B'
            )
            centers[i, :] = res.x

        return centers

    def fit22(self, psf, mask_radius=None):
        """二阶多项式拟合寻心"""
        return self._fit_logic(psf, degree=2, mask_radius=mask_radius)

    def fit44(self, psf, mask_radius=None):
        """四阶多项式拟合寻心"""
        return self._fit_logic(psf, degree=4, mask_radius=mask_radius)

    def fit44_model(self, psf_img, mask_radius):
        poly = PolynomialFeatures(degree=4)
        x_flat, y_flat = self.xx.ravel(), self.yy.ravel()
        # 加入 Mask
        mask = (x_flat ** 2 + y_flat ** 2) <= mask_radius ** 2
        coords = np.column_stack((x_flat[mask], y_flat[mask]))
        features = poly.fit_transform(coords)

        z = psf_img.ravel()[mask]

        model = LinearRegression(fit_intercept=False)
        model.fit(features, z)

        return model, poly