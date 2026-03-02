"""生成多种更醒目的配准对比可视化"""
import cv2
import numpy as np
import os

def read_img(path):
    raw = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    return img

def save_img(path, img, quality=95):
    ext = os.path.splitext(path)[1]
    ok, buf = cv2.imencode(ext, img, [cv2.IMWRITE_PNG_COMPRESSION, 3] if ext == '.png' else [cv2.IMWRITE_JPEG_QUALITY, quality])
    buf.tofile(path)

def split_grid(img, rows=3, cols=2):
    """将2x3网格图拆成6个子图，返回 [(left, right), ...] 三组"""
    h, w = img.shape[:2]
    cell_h = h // rows
    cell_w = w // cols
    pairs = []
    for r in range(rows):
        left = img[r*cell_h:(r+1)*cell_h, 0:cell_w]
        right = img[r*cell_h:(r+1)*cell_h, cell_w:2*cell_w]
        pairs.append((left.copy(), right.copy()))
    return pairs, cell_h, cell_w

# ============ 方案1: 彩色棋盘格 ============
def color_checkerboard(img_a, img_b, block=32):
    """A块染蓝色调，B块染红色调，交替显示"""
    h, w = img_a.shape[:2]
    result = np.zeros_like(img_a)

    # 给A加蓝色调，B加暖色调
    tint_a = img_a.copy().astype(np.float32)
    tint_a[:,:,0] = np.clip(tint_a[:,:,0] * 1.3 + 30, 0, 255)  # B通道增强（蓝）
    tint_a[:,:,2] = np.clip(tint_a[:,:,2] * 0.7, 0, 255)        # R通道减弱
    tint_a = tint_a.astype(np.uint8)

    tint_b = img_b.copy().astype(np.float32)
    tint_b[:,:,2] = np.clip(tint_b[:,:,2] * 1.3 + 30, 0, 255)  # R通道增强（红）
    tint_b[:,:,0] = np.clip(tint_b[:,:,0] * 0.7, 0, 255)        # B通道减弱
    tint_b = tint_b.astype(np.uint8)

    for y in range(0, h, block):
        for x in range(0, w, block):
            ye = min(y + block, h)
            xe = min(x + block, w)
            if ((y // block) + (x // block)) % 2 == 0:
                result[y:ye, x:xe] = tint_a[y:ye, x:xe]
            else:
                result[y:ye, x:xe] = tint_b[y:ye, x:xe]
    return result

# ============ 方案2: 红绿边缘叠加 ============
def edge_overlay(img_a, img_b):
    """提取两图边缘，A染绿色B染红色叠加。对齐处显黄色，不对齐处显单色"""
    gray_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)

    edges_a = cv2.Canny(gray_a, 50, 150)
    edges_b = cv2.Canny(gray_b, 50, 150)

    # 膨胀边缘使其更粗更可见
    kernel = np.ones((2, 2), np.uint8)
    edges_a = cv2.dilate(edges_a, kernel, iterations=1)
    edges_b = cv2.dilate(edges_b, kernel, iterations=1)

    # 底图用灰度
    base = cv2.cvtColor(gray_a, cv2.COLOR_GRAY2BGR)
    base = (base.astype(np.float32) * 0.4).astype(np.uint8)  # 变暗作背景

    # A的边缘 → 绿色, B的边缘 → 红色
    result = base.copy()
    result[edges_a > 0] = [0, 255, 0]    # 绿色 = 原图融合的边缘
    result[edges_b > 0] = [0, 0, 255]    # 红色 = ECC融合的边缘
    # 两者都有的地方 → 黄色（说明对齐）
    both = (edges_a > 0) & (edges_b > 0)
    result[both] = [0, 255, 255]          # 黄色 = 对齐

    return result

# ============ 方案3: 差异热力图 ============
def diff_heatmap(img_a, img_b):
    """计算两图差异，用热力图可视化"""
    gray_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY).astype(np.float32)

    diff = np.abs(gray_a - gray_b)
    diff = cv2.GaussianBlur(diff, (5, 5), 0)
    diff = np.clip(diff * 3, 0, 255).astype(np.uint8)  # 放大差异

    heatmap = cv2.applyColorMap(diff, cv2.COLORMAP_JET)

    # 叠加到原图上
    base = img_b.copy()
    alpha = 0.5
    result = cv2.addWeighted(base, 1 - alpha, heatmap, alpha, 0)

    return result

# ============ 方案4: 滑动对比线 ============
def sliding_compare(img_a, img_b, ratio=0.5):
    """竖向分割线对比，左边A右边B，分割线处加醒目标记"""
    h, w = img_a.shape[:2]
    split_x = int(w * ratio)

    result = img_b.copy()
    result[:, :split_x] = img_a[:, :split_x]

    # 画醒目的分割线
    cv2.line(result, (split_x, 0), (split_x, h), (0, 255, 255), 2)

    # 添加三角形箭头标记
    arrow_y = h // 2
    pts_left = np.array([[split_x - 15, arrow_y - 8], [split_x - 15, arrow_y + 8], [split_x - 3, arrow_y]])
    pts_right = np.array([[split_x + 15, arrow_y - 8], [split_x + 15, arrow_y + 8], [split_x + 3, arrow_y]])
    cv2.fillPoly(result, [pts_left], (0, 200, 255))
    cv2.fillPoly(result, [pts_right], (0, 200, 255))

    return result

# ============ 方案5: 带彩色边框的粗棋盘格 ============
def bold_checkerboard(img_a, img_b, block=48):
    """粗棋盘格 + 彩色粗边框，非常醒目"""
    h, w = img_a.shape[:2]
    result = np.zeros_like(img_a)
    border = 2  # 边框宽度

    for y in range(0, h, block):
        for x in range(0, w, block):
            ye = min(y + block, h)
            xe = min(x + block, w)
            if ((y // block) + (x // block)) % 2 == 0:
                result[y:ye, x:xe] = img_a[y:ye, x:xe]
                # 蓝色边框
                cv2.rectangle(result, (x, y), (xe-1, ye-1), (255, 180, 0), border)
            else:
                result[y:ye, x:xe] = img_b[y:ye, x:xe]
                # 红色边框
                cv2.rectangle(result, (x, y), (xe-1, ye-1), (0, 100, 255), border)
    return result


if __name__ == "__main__":
    src = r"E:\组合卡_result7\result (7).png"
    out_dir = r"E:\组合卡_output"
    os.makedirs(out_dir, exist_ok=True)

    img = read_img(src)
    pairs, cell_h, cell_w = split_grid(img, rows=3, cols=2)

    methods = {
        "A_彩色棋盘格": color_checkerboard,
        "B_红绿边缘叠加": edge_overlay,
        "C_差异热力图": diff_heatmap,
        "D_滑动对比": sliding_compare,
        "E_粗边框棋盘格": bold_checkerboard,
    }

    for name, func in methods.items():
        all_rows = []
        for idx, (left, right) in enumerate(pairs):
            if name == "D_滑动对比":
                vis = func(left, right)
            elif name in ("B_红绿边缘叠加", "C_差异热力图"):
                vis = func(left, right)
            else:
                vis = func(left, right)
            all_rows.append(vis)

        # 重新组合成 3行1列 的图
        combined = np.vstack(all_rows)
        out_path = os.path.join(out_dir, f"{name}.png")
        save_img(out_path, combined)
        print(f"已生成: {out_path}  ({combined.shape[1]}x{combined.shape[0]})")

    # 额外：生成一张总览图，把所有方案横向拼在一起
    print("\n全部完成! 输出目录:", out_dir)
