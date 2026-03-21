# file: check_dataset_safe.py
import pandas as pd
import os

def check_dataset(csv_path):
    if not os.path.exists(csv_path):
        print(f"❌ CSV 文件不存在: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    print(f"✅ CSV 文件加载: {csv_path}")
    print(f"- 总样本数: {len(df)}")

    # 检查 Recording 列
    if 'Recording' not in df.columns:
        print("⚠️ CSV中没有 'Recording' 列")
        return

    # 检查硬负样本
    hard_neg = df[df['Recording'].str.startswith('hardneg_', na=False)]
    print(f"- 硬负样本数量: {len(hard_neg)}")

    # 检查 aug_label 或 label 列
    label_col = 'aug_label' if 'aug_label' in df.columns else ('label' if 'label' in df.columns else None)
    if label_col is None:
        print("⚠️ CSV中没有 'aug_label' 或 'label' 列")
        return

    if not hard_neg.empty:
        print(f"- 硬负样本 {label_col} 范围: {hard_neg[label_col].min()} ~ {hard_neg[label_col].max()}")
        print(f"- 硬负样本前 5 条:")
        print(hard_neg[['Recording', label_col]].head())

    # 检查 NaN
    nan_count = df[label_col].isna().sum()
    print(f"- {label_col} NaN 数量: {nan_count}")

    # 每类样本数量
    print("\n每类样本数量统计:")
    print(df[label_col].value_counts().sort_index())

if __name__ == "__main__":
    csv_path = r"F:\ai-ecg\pretrain_2000_100-100\processed_test\labels.csv" # 替换为你的 CSV 路径
    check_dataset(csv_path)