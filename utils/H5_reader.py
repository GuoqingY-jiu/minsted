"""
HDF5 读取。LPM 组内 ``Dete`` 数据集在本项目中按
``(frames, det, EBP, height, width)`` 解析，空间+通道重排见 ``utils.lpm_h5_layout``。
"""
import h5py
import numpy as np
import os
import json
import tkinter as tk
from tkinter import filedialog
from typing import Optional
from pathlib import Path
from utils.io_utils import select_file


class H5Reader:
    def __init__(self, filepath=None, default_dir=None):
        if filepath:
            self.filepath = filepath
        else:
            if default_dir is None:
                # --- 使用 pathlib 链式跳转 ---
                # .resolve() 获得绝对路径
                # .parent 向上跳一级
                current_path = Path(__file__).resolve()
                project_root = current_path.parent.parent  # 从 utils 跳到 MFX_DL
                default_dir = project_root / "data"  # 自动处理斜杠拼接

            self.filepath = select_file(
                title="选择 HDF5 文件",
                initialdir=default_dir,
                filetypes=[("HDF5 files", "*.h5"), ("All files", "*.*")]
            )

            if not self.filepath:
                raise ValueError("用户取消了文件选择")



    def read_zstack(self,det_index: Optional[int] = None):
        with h5py.File(self.filepath, 'r') as f:
            # 确认根节点包含 raw
            if 'raw' not in f:
                raise KeyError(f"H5 文件 {self.filepath.name} 中未找到 'raw' 数据集")

            raw_dataset = f['raw']
            raw_data = raw_dataset[()]  # 转换为 numpy 数组: (frames, det, height, width)

            # 提取元数据属性
            attr = {}
            for name, value in raw_dataset.attrs.items():
                norm_name = name.replace('(', '').replace(')', '').replace(' ', '_').lower()
                if isinstance(value, bytes):
                    value = value.decode('utf-8')
                attr[norm_name] = value

        # 自动识别探测器通道
        if det_index is None:
            if raw_data.ndim == 4 and raw_data.shape[1] == 2:
                det_index = 1  # 默认取第二路 APD
            else:
                det_index = 0

        # 重排维度: (frames, det, H, W) -> (H, W, 1, frames) 插入虚拟 EBP 通道兼容下游
        if det_index < raw_data.shape[1]:
            vol = raw_data[:, det_index, :, :]
        else:
            vol = np.sum(raw_data, axis=1)

        return vol.transpose(1,2,0), attr


    def read_lpm_data(self, det_index: Optional[int] = None):
        """
        完美合并版：自动处理读取、属性识别与重排
        """
        # 1. 使用改进后的 read_group 获取原始数据
        data_struct, attr = self.read_group('LPM')

        # 2. 这里的 key 已经是 normalize 之后的小写 'dete'
        if not hasattr(data_struct, 'dete'):
            raise KeyError("H5 文件 LPM 组中未找到 'Dete' 数据集")

        raw_data = data_struct.dete

        # 3. 自动识别 det_index (逻辑与你之前的一致)
        _ad = attr.get("active_det", attr.get("lpm_det_index", None))
        if det_index is None:
            if _ad is not None and str(_ad) != "":
                det_index = int(_ad)
            elif raw_data.ndim == 5 and raw_data.shape[1] == 2:
                det_index = 1  # 默认取第二路

        # 4. 调用内部的重排逻辑
        data_struct.dete = self._lpm_dete_to_hwcf(
            raw_data,
            det_index=det_index,
            spatial_axes=str(attr.get("lpm_spatial_axes", "hw")).lower()
        )

        return data_struct, attr

    def read_group(self, group_name):
        if not group_name.startswith('/'):
            group_name = '/' + group_name

        with h5py.File(self.filepath, 'r') as f:
            group = f[group_name]
            attr = self._read_attr(group)
            data = {}
            for key in group.keys():
                item = group[key]
                if isinstance(item, h5py.Dataset):
                    norm_key = self._normalize_name(key)
                    # --- 核心改进：直接读取数值，严禁使用 np.array(item) ---
                    data[norm_key] = item[()]
            return type('DataStruct', (), data), attr

    @staticmethod
    def _lpm_dete_to_hwcf(dete, det_index=None, spatial_axes="hw"):
        """将 (frames, det, EBP, H, W) 转为 (H, W, EBP, frames)"""
        # 强制将 det_index 转为整数，防止触发 numpy 广播
        if det_index is not None:
            det_index = int(det_index)
            vol = dete[:, det_index, :, :, :]
        else:
            vol = np.sum(dete, axis=1)

        if spatial_axes.lower().strip() == "wh":
            vol = np.swapaxes(vol, 2, 3)

        return vol.transpose(2, 3, 1, 0)


    def _read_attr(self, item):
        """内部函数：读取并解析属性，包含 JSON 和 EBP 映射"""
        attr_dict = {}
        for name, value in item.attrs.items():
            norm_name = self._normalize_name(name)

            # 处理 JSON 字符串
            if 'json' in norm_name:
                try:
                    # 如果是 bytes，先转成 string
                    json_str = value.decode('utf-8') if isinstance(value, bytes) else value
                    tmp = json.loads(json_str)
                    for k, v in tmp.items():
                        k_low = k.lower()
                        if 'ebp' in k_low:
                            attr_dict = self._map_ebp(attr_dict, v)
                        else:
                            attr_dict[k_low] = v
                except:
                    attr_dict[norm_name] = value
            else:
                # 处理普通属性
                attr_dict[norm_name] = value
        return attr_dict

    @staticmethod
    def _normalize_name(name):
        """对应 MATLAB 的 normalizeName"""
        return name.replace('(', '').replace(')', '').replace(' ', '_').lower()

    @staticmethod
    def _map_ebp(attr, value):
        """对应 MATLAB 的 mapEBP"""
        ebp_map = {0: 'D4', 1: 'D7', 2: 'D3', 3: 'D6', 4: 'PSF'}
        ebpnum_map = {0: 4, 1: 7, 2: 3, 3: 4, 4: 0}
        if value in ebp_map:
            attr['ebp'] = ebp_map[value]
            attr['ebpnum'] = ebpnum_map[value]
        return attr
if __name__ == "__main__":
    reader = H5Reader()
