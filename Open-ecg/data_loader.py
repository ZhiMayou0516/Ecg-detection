import torch
import numpy as np
import pandas as pd
import os
from torch.utils.data import Dataset
import random


class ECGDataset_Stage1(Dataset):
    """
    Stage 1数据集：支持2k类标签（原始k类 + 硬负样本k类）
    参考：dataset.py中的ECGDataset_unseen
    """

    def __init__(self, phase, data_dir, label_csv, leads=12, length=4096):
        super().__init__()
        self.phase = phase
        self.data_dir = data_dir
        self.leads = leads
        self.length = length

        df = pd.read_csv(label_csv)
        # 筛选对应phase的数据
        df = df[df['split'] == phase]
        self.recordings = df['Recording'].values
        # Stage 1使用aug_label（2k类：0~k-1原始类，k~2k-1硬负样本类）
        self.labels = df['label'].values
        # 保留原始标签用于分析
        self.org_labels = df['org_label'].values if 'org_label' in df.columns else self.labels

    def __len__(self):
        return len(self.labels)

    # 在data_loader.py中，确保__getitem__正确处理形状：
    def __getitem__(self, idx):
        file_name = self.recordings[idx]
        file_path = os.path.join(self.data_dir, file_name)

        # 读取CSV (4096, 12) -> 转置为 (12, 4096)
        df = pd.read_csv(file_path, sep=",", header=None)
        ecg_data = df.values.astype(np.float32)[:self.length, :self.leads]  # (timesteps, leads)

        # ★★★ 添加 NaN 处理 ★★★
        if np.isnan(ecg_data).any():
            print(f"警告：{file_name} 包含 NaN，已用0填充")
            ecg_data = np.nan_to_num(ecg_data, nan=0.0)

        # 长度处理
        nsteps = ecg_data.shape[0]
        if nsteps < self.length:
            result = np.zeros((self.length, self.leads))
            result[-nsteps:, :] = ecg_data
            ecg_data = result
        else:
            ecg_data = ecg_data[-self.length:, :]  # 取最后length个点

        # 转置为 (leads, timesteps) 以符合模型输入 (batch, channels, seq_len)
        ecg_data = ecg_data.transpose()

        # 标签处理（确保不是 NaN）
        label = self.labels[idx]
        if pd.isna(label) or np.isnan(label):
            print(f"警告：索引 {idx} ({file_name}) 的标签为 NaN，设置为原始 org_label")
            label = int(self.org_labels[idx])

        return torch.from_numpy(ecg_data).float(), torch.tensor(int(label), dtype=torch.long)


class ECGDataset_Stage2(Dataset):
    """
    Stage 2数据集：仅包含原始k类，合并train+valid（train_valid）
    参考：dataset.py中的ECGDataset_unseen_MHL_stage2
    """

    def __init__(self, data_dir, label_csv, leads=12, length=4096):
        super().__init__()
        self.data_dir = data_dir
        self.leads = leads
        self.length = length

        df = pd.read_csv(label_csv)
        # Stage 2合并train和valid
        df = df[df['split'].isin(['train', 'valid'])]
        self.recordings = df['Recording'].values
        # Stage 2使用原始label（0~k-1）
        self.labels = df['label'].values

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        file_name = self.recordings[idx]
        file_path = os.path.join(self.data_dir, file_name)
        df = pd.read_csv(file_path, sep=",", header=None)
        ecg_data = df.values.astype(np.float32)[:self.length, :self.leads]

        nsteps = ecg_data.shape[0]
        if nsteps < self.length:
            result = np.zeros((self.length, self.leads))
            result[-nsteps:, :] = ecg_data
            ecg_data = result
        else:
            ecg_data = ecg_data[-self.length:, :]

        label = self.labels[idx]
        return torch.from_numpy(ecg_data.transpose()).float(), torch.tensor(label, dtype=torch.long)


def gen_label_csv_unseen_setting(
        data_dir,
        org_label_csv,
        output_csv,
        unseen_class_idx,
        trn_ratio=0.7,
        val_ratio=0.1,
        seed=42
):
    """
    生成 Stage 1 所需的 CSV：
    - train / valid: 只包含 seen 类
    - test: 包含 seen 类测试样本 + 全部 unseen 类样本
    - Stage 1 的 label 列直接作为最终训练标签：
        seen 原样本 -> 0 ~ k-1
        hard negative -> k ~ 2k-1
        test 原样本 -> 保持原始标签
    """

    import numpy as np
    import pandas as pd

    df = pd.read_csv(org_label_csv).copy()

    if 'Recording' not in df.columns or 'label' not in df.columns:
        raise ValueError("org_label_csv 必须至少包含两列：'Recording' 和 'label'")

    org_y = df['label'].values
    classes = sorted(df['label'].dropna().unique().tolist())

    if unseen_class_idx not in classes:
        raise ValueError(
            f"unseen_class_idx={unseen_class_idx} 不在当前数据标签中，当前标签集合为: {classes}"
        )

    seen_classes = [c for c in classes if c != unseen_class_idx]
    seen_class_num = len(seen_classes)

    print(f"Seen classes: {seen_classes}, count: {seen_class_num}")
    print(f"Unseen class: {unseen_class_idx}")

    # 原始标签 -> 紧凑 seen 标签 0~k-1
    label_map = {old: new for new, old in enumerate(seen_classes)}
    print(f"Label mapping: {label_map}")

    rng = np.random.RandomState(seed)

    trn_idx, val_idx, test_idx = [], [], []

    # 1) seen 类：分别切 train / valid / test
    for cls in seen_classes:
        cls_idx = np.where(org_y == cls)[0]
        cls_idx = cls_idx.copy()
        rng.shuffle(cls_idx)

        n_total = len(cls_idx)

        if n_total == 0:
            continue

        # 尽量保证：
        # - n_total >= 3 时，train / valid / test 都尽量有
        # - n_total == 2 时，train=1, test=1, valid=0
        # - n_total == 1 时，只能放 train
        if n_total == 1:
            n_trn = 1
            n_val = 0
        elif n_total == 2:
            n_trn = 1
            n_val = 0
        else:
            n_trn = max(1, int(round(n_total * trn_ratio)))
            n_val = max(1, int(round(n_total * val_ratio)))

            # 保证至少留 1 个 test
            if n_trn + n_val >= n_total:
                n_val = max(1, n_total - n_trn - 1)
                if n_trn + n_val >= n_total:
                    n_trn = max(1, n_total - n_val - 1)

        cls_trn = cls_idx[:n_trn]
        cls_val = cls_idx[n_trn:n_trn + n_val]
        cls_test = cls_idx[n_trn + n_val:]

        trn_idx.extend(cls_trn.tolist())
        val_idx.extend(cls_val.tolist())
        test_idx.extend(cls_test.tolist())

        print(
            f"Class {cls}: total={n_total}, "
            f"train={len(cls_trn)}, valid={len(cls_val)}, test={len(cls_test)}"
        )

    # 2) unseen 类：全部只放 test
    unseen_idx = np.where(org_y == unseen_class_idx)[0]
    test_idx.extend(unseen_idx.tolist())

    print(f"Unseen samples moved to test: {len(unseen_idx)}")

    # 3) 设置 split
    df['split'] = 'unused'
    df.loc[trn_idx, 'split'] = 'train'
    df.loc[val_idx, 'split'] = 'valid'
    df.loc[test_idx, 'split'] = 'test'

    # 只保留参与实验的样本
    df = df[df['split'].isin(['train', 'valid', 'test'])].copy()

    # 保存原始标签
    df['org_label'] = df['label'].astype(int)

    # 4) train / valid 的 seen 原样本标签映射为 0~k-1
    train_valid_mask = df['split'].isin(['train', 'valid'])
    df.loc[train_valid_mask, 'label'] = df.loc[train_valid_mask, 'org_label'].map(label_map)

    # test 保持原始标签，不映射
    test_mask = df['split'] == 'test'
    df.loc[test_mask, 'label'] = df.loc[test_mask, 'org_label']

    # 5) 为 train / valid 原样本生成 hard negative 记录
    aug_records = []
    base_df = df[df['split'].isin(['train', 'valid'])].copy()

    for _, row in base_df.iterrows():
        compact_label = int(row['label'])   # 此时一定是 0~k-1
        hard_label = compact_label + seen_class_num  # k~2k-1

        new_row = row.copy()
        new_row['Recording'] = f"hardneg_{row['Recording']}"
        new_row['label'] = hard_label
        new_row['aug_label'] = hard_label  # 仅保留做调试，不给 Dataset 主读
        aug_records.append(new_row)

    if len(aug_records) > 0:
        aug_df = pd.DataFrame(aug_records)
        df = pd.concat([df, aug_df], ignore_index=True)
        print(f"Hard negative 标签范围: {aug_df['label'].min()} ~ {aug_df['label'].max()}")

    # 原始样本如果没有 aug_label，统一补 NaN，保持列结构一致
    if 'aug_label' not in df.columns:
        df['aug_label'] = np.nan
    else:
        df.loc[~df['Recording'].astype(str).str.startswith('hardneg_'), 'aug_label'] = np.nan

    # 6) 基本检查
    split_counts = df['split'].value_counts().to_dict()
    print(f"Split counts: {split_counts}")

    train_val_labels = df[df['split'].isin(['train', 'valid'])]['label']
    if len(train_val_labels) == 0:
        raise ValueError("train/valid 为空，无法进行 Stage 1 训练。")

    print(
        f"Stage 1 训练标签范围: "
        f"{train_val_labels.min()} ~ {train_val_labels.max()} "
        f"(应为 0 ~ {2 * seen_class_num - 1})"
    )

    test_labels = df[df['split'] == 'test']['label']
    if len(test_labels) == 0:
        raise ValueError("test 为空，无法进行 open-world 评估。")

    print(f"Test 原始标签集合: {sorted(test_labels.unique().tolist())}")

    df.to_csv(output_csv, index=False)
    print(f"Stage 1 CSV 已保存到: {output_csv}")

    return seen_class_num


def gen_label_csv_stage2(stage1_csv, output_csv, unseen_class_idx):
    """
    生成 Stage 2 所需的 CSV：
    - 只保留 train / valid
    - 只保留原始 seen 样本
    - 排除 hard negative
    - label 保持 Stage 1 中的紧凑 seen 标签 0~k-1
    """
    import pandas as pd

    df = pd.read_csv(stage1_csv).copy()

    required_cols = {'Recording', 'label', 'split', 'org_label'}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"stage1_csv 缺少必要列: {missing_cols}")

    # 1) 只保留 train / valid
    df = df[df['split'].isin(['train', 'valid'])].copy()

    # 2) 排除 unseen 原始类
    df = df[df['org_label'] != unseen_class_idx].copy()

    # 3) 排除 hard negative
    df = df[~df['Recording'].astype(str).str.startswith('hardneg_')].copy()

    if len(df) == 0:
        raise ValueError("Stage 2 CSV 为空，请检查 Stage1 split 或 unseen_class_idx。")

    # 4) label 应该已经是 Stage1 中映射后的 seen 紧凑标签 0~k-1
    if df['label'].isna().any():
        nan_rows = df[df['label'].isna()][['Recording', 'org_label', 'split']]
        raise ValueError(f"Stage 2 CSV 中存在 NaN label，示例行：\n{nan_rows.head()}")

    df['label'] = df['label'].astype(int)
    df['org_label'] = df['org_label'].astype(int)

    unique_labels = sorted(df['label'].unique().tolist())
    expected_labels = list(range(len(unique_labels)))

    # 不要静默重映射，直接检查是否连续
    if unique_labels != expected_labels:
        raise ValueError(
            f"Stage 2 label 不连续，当前为 {unique_labels}，期望类似 {expected_labels}。\n"
            f"这通常说明 Stage1 的标签映射或划分仍有问题。"
        )

    # 5) 保存
    df.to_csv(output_csv, index=False)

    # 6) 统计
    class_counts = df['label'].value_counts().sort_index()
    print(f"Stage 2 CSV 已保存到: {output_csv}")
    print("Stage 2 split 统计:", df['split'].value_counts().to_dict())
    print("Stage 2 label 集合:", unique_labels)
    print("Stage 2 各类样本数:")
    print(class_counts)

    return class_counts.tolist()