import torch
import argparse
import os
import h5py
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from data_loader import (ECGDataset_Stage1, ECGDataset_Stage2,
                         gen_label_csv_unseen_setting, gen_label_csv_stage2)
from model import resnet18_model, resnet34_MHL
# 在 main(1).py 顶部添加导入
from train import train_stage1, train_stage2, load_stage1_weights_into_mhl, extract_prototypes
# 在Stage 1之前添加硬负样本生成步骤
from generate_hard_negatives import HardNegativeGenerator


from evaluation import compute_prototypes_and_stats, evaluate_open_world


def load_hdf5_weights(model, hdf5_path):
    """加载 HDF5 格式权重，并打印详细匹配情况（含 BN buffers）"""
    import os
    import h5py
    import torch

    if not os.path.exists(hdf5_path):
        print(f"权重文件不存在: {hdf5_path}")
        return model

    print(f"正在加载 HDF5 权重: {hdf5_path}")

    with h5py.File(hdf5_path, "r") as f:
        print("HDF5 文件结构:")
        f.visititems(lambda name, obj: print(name))

        state_dict = model.state_dict()
        loaded_keys = []
        missing_keys = []
        shape_mismatch = []

        new_state = {}

        for name, tensor in state_dict.items():
            if name not in f:
                missing_keys.append(name)
                continue

            data = torch.from_numpy(f[name][:]).float()

            if tuple(data.shape) != tuple(tensor.shape):
                shape_mismatch.append(
                    f"{name}: model {tuple(tensor.shape)} vs file {tuple(data.shape)}"
                )
                continue

            new_state[name] = data
            loaded_keys.append(name)

        # 用 strict=False 更新
        state_dict.update(new_state)
        model.load_state_dict(state_dict, strict=False)

    print(f"\n✅ 成功加载 {len(loaded_keys)} 个张量")
    if loaded_keys:
        print("前 20 个成功加载层：")
        for k in loaded_keys[:20]:
            print("  ✓", k)

    print(f"\n⚠️ 缺失 {len(missing_keys)} 个张量")
    if missing_keys:
        for k in missing_keys[:20]:
            print("  ✗", k)

    print(f"\n⚠️ shape 不匹配 {len(shape_mismatch)} 个张量")
    if shape_mismatch:
        for msg in shape_mismatch[:30]:
            print("  !", msg)

    return model


def load_pretrained_weights(model, npy_path, device='cuda'):
    """
    加载 Nature 论文的预训练权重（兼容多种格式）
    """
    if not os.path.exists(npy_path):
        print(f"⚠️  预训练权重不存在，使用随机初始化")
        return model

    print(f"正在尝试加载: {npy_path}")

    try:
        # 加载文件
        data = np.load(npy_path, allow_pickle=True)

        # 检查数据类型
        print(f"文件类型: {type(data)}, 形状: {getattr(data, 'shape', 'N/A')}")

        # 情况 1: 如果是 numpy array 且是 0 维或包含单个对象
        if isinstance(data, np.ndarray) and data.shape == ():
            pretrained_dict = data.item()
        # 情况 2: 如果是 numpy array 包含字典
        elif isinstance(data, np.ndarray) and len(data.shape) == 1:
            pretrained_dict = data[0] if isinstance(data[0], dict) else data
        # 情况 3: 已经是字典
        elif isinstance(data, dict):
            pretrained_dict = data
        else:
            pretrained_dict = data.item() if hasattr(data, 'item') else data

        print(f"权重字典包含 {len(pretrained_dict)} 个条目")
        print(f"示例键: {list(pretrained_dict.keys())[:5]}...")

    except Exception as e:
        print(f"❌ 无法解析权重文件: {e}")
        print("将使用随机初始化继续...")
        return model

    # 加载到模型
    model_dict = model.state_dict()
    compatible_dict = {}
    incompatible_keys = []

    for k, v in pretrained_dict.items():
        # 转换键名格式（TensorFlow→PyTorch 常见映射）
        pytorch_key = k
        if '/' in k:  # TF 格式通常有 '/' 分隔
            pytorch_key = k.replace('/', '.')

        if pytorch_key in model_dict:
            try:
                # 确保是 numpy 数组
                if not isinstance(v, np.ndarray):
                    v = np.array(v)

                # 检查形状匹配
                if model_dict[pytorch_key].shape == v.shape:
                    compatible_dict[pytorch_key] = torch.from_numpy(v).float()
                else:
                    incompatible_keys.append(f"{pytorch_key}: 模型{model_dict[pytorch_key].shape} vs 文件{v.shape}")
            except Exception as e:
                incompatible_keys.append(f"{pytorch_key}: {str(e)}")
        else:
            incompatible_keys.append(f"{k}: 键名不匹配")

    # 更新模型
    if compatible_dict:
        model_dict.update(compatible_dict)
        model.load_state_dict(model_dict, strict=False)
        print(f"✅ 成功加载 {len(compatible_dict)}/{len(model_dict)} 层")

        if incompatible_keys:
            print(f"⚠️  跳过 {len(incompatible_keys)} 层（键名或形状不匹配）")
    else:
        print("⚠️  没有匹配的层可加载，使用随机初始化")
        print(f"调试信息 - 文件中的键示例: {list(pretrained_dict.keys())[:10]}")
        print(f"调试信息 - 模型中的键示例: {list(model_dict.keys())[:10]}")

    return model

def main():
    parser = argparse.ArgumentParser(description="AI+ECG Open World Training")
    # 数据参数
    parser.add_argument('--data_dir', type=str, required=True, help='ECG数据目录')
    parser.add_argument('--label_csv', type=str, required=True, help='原始标签CSV')
    parser.add_argument('--output_dir', type=str, default='./output', help='输出目录')
    parser.add_argument('--unseen_class_idx', type=int, default=4, help='Unseen类索引')

    # 训练参数
    parser.add_argument('--epochs_stage1', type=int, default=200, help='Stage 1轮数')
    parser.add_argument('--epochs_stage2', type=int, default=200, help='Stage 2轮数')
    parser.add_argument('--lr', type=float, default=0.0001, help='Stage 1学习率')
    parser.add_argument('--lr_stage2', type=float, default=0.0001, help='Stage 2学习率')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--alpha', type=float, default=0.6, help='SupConLoss权重')
    parser.add_argument('--temp', type=float, default=0.1, help='对比学习温度')
    parser.add_argument('--gamma', type=float, default=2.0, help='开放世界阈值系数')
    parser.add_argument('--pretrained_path', type=str, default=None,
                        help='Nature论文预训练权重路径 (model.npy)')
    parser.add_argument('--freeze_backbone', action='store_true',
                        help='是否冻结特征提取层（只训练分类头）')

    parser.add_argument('--eval_only', action='store_true', help='只做评估，跳过Stage1/Stage2训练')
    parser.add_argument('--stage2_ckpt', type=str, default=None, help='Stage2模型权重路径(.pth)')

    args = parser.parse_args()
    args.save_dir = args.output_dir
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    generator = HardNegativeGenerator(sampling_rate=400)
    generator.batch_generate(args.label_csv, args.data_dir, args.data_dir)

    # 1. 生成Stage 1标签（包含硬负样本，2k类）
    stage1_csv = os.path.join(args.output_dir, 'labels_stage1.csv')
    seen_class_num = gen_label_csv_unseen_setting(
        args.data_dir, args.label_csv, stage1_csv,
        args.unseen_class_idx, seed=42
    )
    tmp_df = pd.read_csv(stage1_csv)
    print(tmp_df['split'].value_counts())
    print(tmp_df.groupby('split')['label'].apply(lambda x: sorted(pd.unique(x))).to_dict())
    print(f"Seen classes: {seen_class_num}, Total Stage 1 classes: {seen_class_num * 2}")
    print("\n===== VALID label count =====")
    print(tmp_df[tmp_df["split"] == "valid"]["label"].value_counts().sort_index())

    print("\n===== TRAIN label count =====")
    print(tmp_df[tmp_df["split"] == "train"]["label"].value_counts().sort_index())

    print("\n===== TEST original label count =====")
    print(tmp_df[tmp_df["split"] == "test"]["label"].value_counts().sort_index())
    base_seen_df = tmp_df[
        tmp_df['split'].isin(['train', 'valid']) &
        (~tmp_df['Recording'].astype(str).str.startswith('hardneg_'))
        ].copy()

    seen_label_map = (
        base_seen_df[['org_label', 'label']]
        .drop_duplicates()
        .sort_values('org_label')
    )

    seen_label_map = dict(zip(seen_label_map['org_label'].astype(int),
                              seen_label_map['label'].astype(int)))

    print("Seen label map:", seen_label_map)
    stage2_csv = os.path.join(args.output_dir, 'labels_stage2.csv')

    if args.eval_only:
        # 如果 stage2_csv 不存在，就基于已有的 stage1_csv 生成
        if not os.path.exists(stage2_csv):
            gen_label_csv_stage2(stage1_csv, stage2_csv, args.unseen_class_idx)

        # Stage2 训练集（用于计算 prototype / mean_dist / std_dist）
        train_dataset_proto = ECGDataset_Stage2(args.data_dir, stage2_csv)
        train_loader_proto = DataLoader(train_dataset_proto, batch_size=args.batch_size, shuffle=False)

        # 加载 Stage2 模型
        ckpt_path = args.stage2_ckpt if args.stage2_ckpt is not None else os.path.join(args.output_dir,
                                                                                       'stage2_best.pth')
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"找不到 Stage2 权重文件: {ckpt_path}")

        model_s2 = resnet34_MHL().to(device)
        model_s2.load_state_dict(torch.load(ckpt_path, map_location=device))
        model_s2.eval()

        print(f"已加载 Stage2 权重: {ckpt_path}")

        # 测试集
        test_dataset = ECGDataset_Stage1('test', args.data_dir, stage1_csv)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        # 用已加载的 Stage2 模型重新计算 prototype 和统计量
        prototypes_s2, mean_dists, std_dists = compute_prototypes_and_stats(
            model_s2, train_loader_proto, seen_class_num, device
        )

        preds, labels, report = evaluate_open_world(
            model_s2, test_loader, prototypes_s2, mean_dists, std_dists,
            unseen_class_idx=args.unseen_class_idx,
            seen_class_num=seen_class_num,
            seen_label_map=seen_label_map,
            gamma=args.gamma,
            device=device
        )

        print("Open World Evaluation Completed!")
        print(f"Macro F1: {report['macro avg']['f1-score']:.4f}")
        return

    # 2. Stage 1训练（监督对比学习）
    train_dataset_s1 = ECGDataset_Stage1('train', args.data_dir, stage1_csv)
    val_dataset_s1 = ECGDataset_Stage1('valid', args.data_dir, stage1_csv)
    train_loader_s1 = DataLoader(train_dataset_s1, batch_size=args.batch_size, shuffle=True)
    val_loader_s1 = DataLoader(val_dataset_s1, batch_size=args.batch_size, shuffle=False)

    print(f"创建 Stage 1 模型 (num_classes={seen_class_num * 2})...")
    model_s1 = resnet18_model(num_classes=seen_class_num * 2).to(device)

    # ★★★ 在这里加载预训练权重 ★★★
    if args.pretrained_path and args.pretrained_path.endswith(('.h5','.hdf5')):
        model_s1 = load_hdf5_weights(model_s1, args.pretrained_path)

        # 可选：冻结特征提取层（如果数据量小）
        if args.freeze_backbone:
            print("冻结特征提取层（conv1, res_unit*, fc）...")
            for name, param in model_s1.named_parameters():
                # 冻结除 fc2 外的所有层（fc2是分类头，需要训练2k类输出）
                if 'fc2' not in name:
                    param.requires_grad = False
                else:
                    print(f"可训练: {name}")

    # 继续原有训练流程
    stage1_path = train_stage1(model_s1, train_loader_s1, val_loader_s1, args)

    # 3. 提取Prototypes（使用Stage 1训练好的模型）
    # 使用Stage 2的数据加载方式（只含原始k类）来提取prototype
    stage2_csv = os.path.join(args.output_dir, 'labels_stage2.csv')
    # ★★★ 关键修复：传入 stage1_csv（包含 split 列），而不是 args.label_csv ★★★
    gen_label_csv_stage2(stage1_csv, stage2_csv, args.unseen_class_idx)  # 修改这里

    train_dataset_proto = ECGDataset_Stage2(args.data_dir, stage2_csv)
    train_loader_proto = DataLoader(train_dataset_proto, batch_size=args.batch_size)

    prototypes = extract_prototypes(model_s1, train_loader_proto, seen_class_num)

    # 4. Stage 2训练（多超球面学习）
    train_dataset_s2 = ECGDataset_Stage2(args.data_dir, stage2_csv)  # 自动合并train+valid
    train_loader_s2 = DataLoader(train_dataset_s2, batch_size=args.batch_size, shuffle=True)

    model_s2 = resnet34_MHL().to(device)  # 使用MHL专用模型
    model_s2 = load_stage1_weights_into_mhl(stage1_path, model_s2)
    stage2_path = train_stage2(model_s2, train_loader_s2, prototypes, args)
    model_s2.load_state_dict(torch.load(stage2_path, map_location=device))
    model_s2.eval()

    # 5. 开放世界评估
    test_dataset = ECGDataset_Stage1('test', args.data_dir, stage1_csv)  # 测试集包含unseen
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # 重新计算prototypes和统计量（使用Stage 2模型）
    prototypes_s2, mean_dists, std_dists = compute_prototypes_and_stats(
        model_s2, train_loader_proto, seen_class_num, device
    )

    preds, labels, report = evaluate_open_world(
        model_s2, test_loader, prototypes_s2, mean_dists, std_dists,
        unseen_class_idx=args.unseen_class_idx,
        seen_class_num=seen_class_num,
        seen_label_map=seen_label_map,
        gamma=args.gamma,
        device=device
    )

    print("Open World Evaluation Completed!")
    print(f"Macro F1: {report['macro avg']['f1-score']:.4f}")


if __name__ == "__main__":
    main()