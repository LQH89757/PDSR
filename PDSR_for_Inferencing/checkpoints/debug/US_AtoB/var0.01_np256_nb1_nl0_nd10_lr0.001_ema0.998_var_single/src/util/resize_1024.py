import os
import cv2

root_dir = r"../results/debug/US_AtoB/var0.01_np256_nb1_nl0_nd10_lr0.001_ema0.998_var_single/test_latest/images"

# 目标比例
target_ratio = 7011 / 5494
long_side = 1024

for folder in os.listdir(root_dir):
    folder_path = os.path.join(root_dir, folder)
    if not os.path.isdir(folder_path):
        continue

    for file in os.listdir(folder_path):
        if not file.lower().endswith(".png"):
            continue

        img_path = os.path.join(folder_path, file)
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

        if img is None:
            print(f"读取失败: {img_path}")
            continue

        h, w = img.shape[:2]

        # 等比例缩放，保持长边为 1024
        if target_ratio >= 1:
            # 宽为长边
            new_w = long_side
            new_h = int(long_side / target_ratio)
        else:
            # 高为长边（理论上不会发生）
            new_h = long_side
            new_w = int(long_side * target_ratio)

        resized = cv2.resize(
            img,
            (new_w, new_h),
            interpolation=cv2.INTER_CUBIC
        )

        # 覆盖保存（如需另存可改路径）
        cv2.imwrite(img_path, resized)

print("所有图像已按 7011×5494 比例缩放，且长边保持为 1024。")
