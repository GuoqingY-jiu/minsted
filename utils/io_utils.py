import os
import tkinter as tk
from tkinter import filedialog


def select_file(title="选择文件", initialdir=None, filetypes=None, multiple=False):
    """
    弹出文件选择对话框并返回路径。

    参数:
    - title: 窗口标题
    - initialdir: 初始打开目录
    - filetypes: 允许的文件类型列表，例如 [("PyTorch Files", "*.pt"), ("All Files", "*.*")]
    - multiple: False(默认) 返回 str; True 返回 tuple (多选)
    """
    # 如果初始路径不存在，设为当前工作目录，防止函数报错
    if initialdir and not os.path.exists(initialdir):
        initialdir = os.getcwd()

    root = tk.Tk()
    root.overrideredirect(True)  # 移除窗口边框
    root.attributes('-alpha', 0)  # 设置完全透明
    root.geometry("0x0+0+0")  # 缩到最小放在屏幕角落
    root.lift()
    root.attributes('-topmost', True)  # 强制对话框置顶，防止被 IDE 窗口遮挡
    all_selected_files = []
    current_dir = initialdir

    if multiple:
        while True:
            file_path = filedialog.askopenfilenames(
                parent=root,
                title=f"{title} (已选 {len(all_selected_files)} 个，取消或关闭以结束选择)",
                initialdir=current_dir,
                filetypes=filetypes if filetypes else [("All Files", "*.*")]
            )

            if not file_path:  # 用户点取消或叉号，跳出循环
                break
            all_selected_files.extend(file_path)
            # 更新下一次打开的目录为最后一次选择的目录
            current_dir = os.path.dirname(file_path[0])
            print(f"当前已累计选择 {len(all_selected_files)} 个文件...")
        selected_file = list(dict.fromkeys(all_selected_files))

    else:
        selected_file = filedialog.askopenfilename(
            parent=root,
            title=title,
            initialdir=initialdir,
            filetypes=filetypes if filetypes else [("All Files", "*.*")]
        )

    root.destroy()

    if not selected_file:
        return None

    return selected_file