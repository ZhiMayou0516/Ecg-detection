import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import shutil
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import resnet18_model, ResNet1D_MHL
from loss_func import SupConLoss, MultiHypersphereLoss
from data_loader import ECGDataset_Stage1, ECGDataset_Stage2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def euclidean_distance(node_embedding, c):
    """计算节点到原型的欧氏距离平方"""
    return torch.sum((node_embedding - c) ** 2, dim=1)


def train_stage1(model, train_loader, val_loader, args):
    train_log = []
    criterion = SupConLoss(alpha=args.alpha, temperature=args.temp)
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs_stage1
    )

    # 用 val_ce 选 best，比 val_total 更稳
    best_val_ce = float("inf")
    model_path = os.path.join(args.save_dir, "stage1_best.pth")

    for epoch in range(args.epochs_stage1):
        # ==================== train ====================
        model.train()
        train_total_sum = 0.0
        train_ce_sum = 0.0
        train_cl_sum = 0.0
        train_correct = 0
        train_count = 0

        for data, labels in tqdm(train_loader, desc=f"Stage1 Epoch {epoch + 1}"):
            data, labels = data.to(device), labels.to(device)

            hidden_emb, logits = model.get_embedding_and_logits(data)

            # 分开计算 CE / CL
            ce_raw = criterion.ce_loss(logits, labels)
            cl_raw = criterion.nt_xent_loss(hidden_emb, labels)

            ce_loss = (1 - args.alpha) * ce_raw
            cl_loss = args.alpha * cl_raw
            loss = ce_loss + cl_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            bs = labels.size(0)
            train_total_sum += loss.item() * bs
            train_ce_sum += ce_loss.item() * bs
            train_cl_sum += cl_loss.item() * bs

            preds = torch.argmax(logits, dim=1)
            train_correct += (preds == labels).sum().item()
            train_count += bs

        avg_train_total = train_total_sum / max(train_count, 1)
        avg_train_ce = train_ce_sum / max(train_count, 1)
        avg_train_cl = train_cl_sum / max(train_count, 1)
        train_acc = train_correct / max(train_count, 1)

        # ==================== val ====================
        model.eval()
        val_total_sum = 0.0
        val_ce_sum = 0.0
        val_cl_sum = 0.0
        val_correct = 0
        val_count = 0

        # 新增：统计整个验证集的真实标签 / 预测标签分布
        val_label_hist = None
        val_pred_hist = None

        # 新增：保存第一个 batch 的 labels / preds，方便直接看是否塌成单类
        first_val_batch_labels = None
        first_val_batch_preds = None

        with torch.no_grad():
            for batch_idx, (data, labels) in enumerate(val_loader):
                data, labels = data.to(device), labels.to(device)

                hidden_emb, logits = model.get_embedding_and_logits(data)

                ce_raw = criterion.ce_loss(logits, labels)
                cl_raw = criterion.nt_xent_loss(hidden_emb, labels)

                ce_loss = (1 - args.alpha) * ce_raw
                cl_loss = args.alpha * cl_raw
                val_loss = ce_loss + cl_loss

                bs = labels.size(0)
                val_total_sum += val_loss.item() * bs
                val_ce_sum += ce_loss.item() * bs
                val_cl_sum += cl_loss.item() * bs

                preds = torch.argmax(logits, dim=1)
                val_correct += (preds == labels).sum().item()
                val_count += bs

                # ========== 统计 label / pred 分布 ==========
                num_classes = logits.size(1)

                batch_label_hist = torch.bincount(labels.detach().cpu(), minlength=num_classes)
                batch_pred_hist = torch.bincount(preds.detach().cpu(), minlength=num_classes)

                if val_label_hist is None:
                    val_label_hist = batch_label_hist
                    val_pred_hist = batch_pred_hist
                else:
                    val_label_hist += batch_label_hist
                    val_pred_hist += batch_pred_hist

                # 只保存第一个 batch 用来检查
                if batch_idx == 0:
                    first_val_batch_labels = labels.detach().cpu().tolist()
                    first_val_batch_preds = preds.detach().cpu().tolist()

        avg_val_total = val_total_sum / max(val_count, 1)
        avg_val_ce = val_ce_sum / max(val_count, 1)
        avg_val_cl = val_cl_sum / max(val_count, 1)
        val_acc = val_correct / max(val_count, 1)

        scheduler.step()

        print(
            f"Stage 1 Epoch {epoch + 1} | "
            f"Train Total: {avg_train_total:.4f} (CE: {avg_train_ce:.4f}, CL: {avg_train_cl:.4f}, Acc: {train_acc:.4f}) | "
            f"Val Total: {avg_val_total:.4f} (CE: {avg_val_ce:.4f}, CL: {avg_val_cl:.4f}, Acc: {val_acc:.4f})"
        )

        # ===== 新增调试输出：前5个epoch都打印 =====
        if epoch < 5:
            print(f"\n[Epoch {epoch + 1}] VALID label histogram:")
            print(val_label_hist.tolist() if val_label_hist is not None else None)

            print(f"[Epoch {epoch + 1}] VALID pred histogram:")
            print(val_pred_hist.tolist() if val_pred_hist is not None else None)

            print(f"[Epoch {epoch + 1}] First val batch labels:")
            print(first_val_batch_labels)

            print(f"[Epoch {epoch + 1}] First val batch preds:")
            print(first_val_batch_preds)
            print("")

        train_log.append({
            "epoch": epoch + 1,
            "train_total": avg_train_total,
            "train_ce": avg_train_ce,
            "train_cl": avg_train_cl,
            "train_acc": train_acc,
            "val_total": avg_val_total,
            "val_ce": avg_val_ce,
            "val_cl": avg_val_cl,
            "val_acc": val_acc,
            "lr": optimizer.param_groups[0]["lr"],
        })

        # 关键：按 val_ce 保存，而不是 val_total
        if avg_val_ce < best_val_ce:
            best_val_ce = avg_val_ce
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_ce": best_val_ce,
            }, model_path)
            print(f"  -> Saved best Stage 1 model at epoch {epoch + 1} (best val_ce={best_val_ce:.4f})")
    pd.DataFrame(train_log).to_csv(
        os.path.join(args.output_dir, "train_stage1_log.csv"), index=False
    )
    print(f"✅ Stage1训练日志已保存到 {args.output_dir}/train_stage1_log.csv")
    return model_path


def extract_prototypes(model, data_loader, seen_class_num):
    """
    提取训练集embedding并计算各类原型（prototype）
    参考：run_1_train.py中Stage 2前的prototype计算
    """
    model.eval()
    embeddings_dict = {i: [] for i in range(seen_class_num)}

    with torch.no_grad():
        for data, labels in tqdm(data_loader, desc='Extracting Prototypes'):
            data = data.to(device)
            # 使用ResNet1D的get_embedding或Stage1模型的前向传播
            embs = model.get_embedding(data) if hasattr(model, 'get_embedding') else model(data)
            embs = embs.cpu().numpy()
            labels = labels.numpy()

            for i in range(len(labels)):
                label = labels[i]
                if label < seen_class_num:  # 只收集原始k类（排除硬负样本）
                    embeddings_dict[label].append(embs[i])

    # 计算每个类的均值作为原型
    prototypes = []
    for i in range(seen_class_num):
        if len(embeddings_dict[i]) > 0:
            proto = np.mean(embeddings_dict[i], axis=0)
            prototypes.append(torch.from_numpy(proto).float().to(device))
        else:
            prototypes.append(torch.zeros(30).to(device))  # 假设embedding dim=30
    return prototypes


def train_stage2(model, train_loader, prototypes, args):
    """
    Stage 2: 多超球面学习（Multi-Hypersphere Learning）
    参考：run_1_train.py中的run_MHL函数
    """
    train_log_stage2 = []
    criterion = MultiHypersphereLoss()
    optimizer = Adam(model.parameters(), lr=args.lr_stage2)

    # 将prototypes转为tensor并固定（不参与梯度）
    proto_tensors = [p.clone().detach() for p in prototypes]

    best_loss = float('inf')
    model_path = os.path.join(args.save_dir, 'stage2_best.pth')

    for epoch in range(args.epochs_stage2):
        model.train()
        running_loss = 0.0  # 训练目标：平方距离均值
        running_mean_dist = 0.0  # 额外日志：欧氏距离均值
        running_emb_norm = 0.0  # 额外日志：embedding范数均值
        steps = 0

        for data, labels in tqdm(train_loader, desc=f'Stage2 Epoch {epoch + 1}'):
            data, labels = data.to(device), labels.to(device)
            embeddings = model(data)  # [B, 30]

            # ===== batch级统计 =====
            valid_classes = 0
            batch_mean_dist_sum = 0.0
            batch_emb_norm_sum = 0.0

            for j in range(len(proto_tensors)):
                mask = (labels == j)
                if mask.sum() == 0:
                    continue

                class_embs = embeddings[mask]

                # 只用于日志统计，不再手写loss
                distances = euclidean_distance(class_embs, proto_tensors[j])  # [n_j]

                # 1) 额外打印用：真实欧氏距离均值
                class_euclid_mean = torch.sqrt(distances + 1e-12).mean()

                # 2) 额外打印用：embedding范数均值
                class_emb_norm_mean = torch.norm(class_embs, dim=1).mean()

                batch_mean_dist_sum += class_euclid_mean.item()
                batch_emb_norm_sum += class_emb_norm_mean.item()
                valid_classes += 1

            if valid_classes == 0:
                continue

            loss = criterion(embeddings, labels, proto_tensors)

            # 日志指标：各有效类别欧氏距离均值、embedding范数均值
            batch_mean_dist = batch_mean_dist_sum / valid_classes
            batch_emb_norm = batch_emb_norm_sum / valid_classes

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_mean_dist += batch_mean_dist
            running_emb_norm += batch_emb_norm
            steps += 1

        avg_loss = running_loss / max(steps, 1)
        avg_mean_dist = running_mean_dist / max(steps, 1)
        avg_emb_norm = running_emb_norm / max(steps, 1)

        print(
            f"Stage 2 Epoch {epoch + 1}, "
            f"Loss(sqdist): {avg_loss:.4f}, "
            f"MeanDist(euclid): {avg_mean_dist:.4f}, "
            f"EmbNorm: {avg_emb_norm:.4f}"
        )

        train_log_stage2.append({
            "epoch": epoch + 1,
            "train_loss_sqdist": avg_loss,
            "train_mean_dist": avg_mean_dist,
            "train_emb_norm": avg_emb_norm
        })

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), model_path)
            print(f"  -> Saved best Stage 2 model")

    pd.DataFrame(train_log_stage2).to_csv(
        os.path.join(args.output_dir, "train_stage2_log.csv"),
        index=False
    )
    print(f"✅ Stage2训练日志已保存到 {args.output_dir}/train_stage2_log.csv")
    return model_path


def load_stage1_weights_into_mhl(stage1_path, mhl_model):
    """
    将Stage 1的权重加载到Stage 2模型（跳过最后的分类层）
    参考：run_1_train.py中的load_checkpoints
    """
    checkpoint = torch.load(stage1_path, map_location=device)
    state_dict = checkpoint['model_state_dict']

    # 过滤掉Stage 1特有的层（如最后的fc层）
    mhl_dict = mhl_model.state_dict()
    filtered_dict = {k: v for k, v in state_dict.items() if k in mhl_dict and v.shape == mhl_dict[k].shape}
    mhl_model.load_state_dict(filtered_dict, strict=False)
    return mhl_model


# 需要修改ResNet1D以支持返回embedding和logits
# 在model.py中添加以下方法到ResNet1D类：
def get_embedding_and_logits(self, x):
    """同时返回embedding和logits，用于Stage 1"""
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.res_unit1(x)
    x = self.res_unit2(x)
    x = self.res_unit3(x)
    x = self.res_unit4(x)
    x = self.flatten(x)
    embedding = self.fc(x)  # 假设fc输出30维，然后接一个fc2输出logits
    logits = self.fc2(embedding) if hasattr(self, 'fc2') else embedding
    return embedding, logits