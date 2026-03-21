import torch
import numpy as np
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix
import warnings


def evaluate_model(model, dataloader, device, threshold=0.5):
    """
    多标签分类评估（严格对应论文 Table 2 指标）
    计算每个类别的 Precision (PPV), Recall (Sensitivity), Specificity, F1
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for data, labels in dataloader:
            data = data.to(device)
            outputs = model(data)  # logits
            probs = torch.sigmoid(outputs)  # 概率
            preds = (probs > threshold).float()

            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    all_probs = np.vstack(all_probs)

    # 计算每个类别的指标（多标签）
    target_names = ["1dAVb", "RBBB", "LBBB", "SB", "AF", "ST"]

    print("Classification Report (per class):")
    print(classification_report(all_labels, all_preds,
                                target_names=target_names, digits=4))

    # 计算宏平均 F1 (与论文一致)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    micro_f1 = f1_score(all_labels, all_preds, average='micro')

    # 计算 Specificity (TN/(TN+FP))
    specificity = []
    for i in range(len(target_names)):
        tn = np.sum((all_labels[:, i] == 0) & (all_preds[:, i] == 0))
        fp = np.sum((all_labels[:, i] == 0) & (all_preds[:, i] == 1))
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        specificity.append(spec)

    print(f"\nMacro F1: {macro_f1:.4f}")
    print(f"Micro F1: {micro_f1:.4f}")
    print("Specificity per class:", dict(zip(target_names, [f"{s:.4f}" for s in specificity])))

    return {
        'preds': all_preds,
        'labels': all_labels,
        'probs': all_probs,
        'macro_f1': macro_f1,
        'specificity': specificity
    }


import torch
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, precision_recall_fscore_support
from scipy.spatial.distance import euclidean


def compute_prototypes_and_stats(model, dataloader, seen_class_num, device):
    """
    计算各类原型（prototype）和距离统计量（mean, std）
    参考：run_2_results.py中的compute_distances
    """
    model.eval()
    embeddings_dict = {i: [] for i in range(seen_class_num)}

    with torch.no_grad():
        for data, labels in dataloader:
            data = data.to(device)
            embeddings = model(data).cpu().numpy()
            labels = labels.numpy()

            for emb, label in zip(embeddings, labels):
                if label < seen_class_num:
                    embeddings_dict[label].append(emb)

    prototypes = []
    mean_distances = []
    std_distances = []

    for i in range(seen_class_num):
        if len(embeddings_dict[i]) == 0:
            prototypes.append(np.zeros(30))
            mean_distances.append(0)
            std_distances.append(1)
            continue

        proto = np.mean(embeddings_dict[i], axis=0)
        prototypes.append(proto)

        # 计算类内距离统计量
        distances = [euclidean(proto, emb) for emb in embeddings_dict[i]]
        mean_distances.append(np.mean(distances))
        std_distances.append(np.std(distances))

    return prototypes, mean_distances, std_distances


def evaluate_open_world(model, dataloader, prototypes, mean_dists, std_dists,
                        unseen_class_idx, seen_class_num, seen_label_map,
                        gamma=2.0, device='cuda'):
    from sklearn.metrics import classification_report
    from scipy.spatial.distance import euclidean
    import numpy as np
    import torch

    model.eval()
    all_preds = []
    all_labels = []

    unseen_eval_idx = seen_class_num  # 最终评估空间里，最后一个编号专门给 unseen

    with torch.no_grad():
        for data, labels in dataloader:
            data = data.to(device)
            embeddings = model(data).cpu().numpy()
            true_labels = labels.numpy()

            for emb, true_label in zip(embeddings, true_labels):
                # ---------- prediction ----------
                distances = [euclidean(emb, proto) for proto in prototypes]
                min_dist = min(distances)
                pred_class = distances.index(min_dist)   # 0 ~ seen_class_num-1

                threshold = mean_dists[pred_class] + gamma * std_dists[pred_class]

                if min_dist > threshold:
                    pred = unseen_eval_idx
                else:
                    pred = pred_class

                # ---------- true label ----------
                true_label = int(true_label)

                if true_label == unseen_class_idx:
                    mapped_true = unseen_eval_idx
                elif true_label in seen_label_map:
                    mapped_true = seen_label_map[true_label]
                else:
                    raise ValueError(f"测试集中发现无法映射的原始标签: {true_label}")

                all_preds.append(pred)
                all_labels.append(mapped_true)

    if len(all_labels) == 0:
        raise ValueError("测试集为空：evaluate_open_world 没有收到任何样本。")

    eval_labels = list(range(seen_class_num + 1))
    target_names = [f'Class_{i}' for i in range(seen_class_num)] + ['Unseen']

    report = classification_report(
        all_labels,
        all_preds,
        labels=eval_labels,
        target_names=target_names,
        output_dict=True,
        zero_division=0
    )

    print(classification_report(
        all_labels,
        all_preds,
        labels=eval_labels,
        target_names=target_names,
        zero_division=0
    ))

    return all_preds, all_labels, report


def evaluate_closed_world(model, dataloader, device):
    """
    标准封闭世界评估（用于Baseline对比）
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data, labels in dataloader:
            data = data.to(device)
            outputs = model(data)
            if isinstance(outputs, tuple):
                outputs = outputs[1]  # 如果返回的是(embedding, logits)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    # 计算多标签指标...
    return np.vstack(all_preds), np.vstack(all_labels)