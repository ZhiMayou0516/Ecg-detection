import os
from typing import Dict, List, Tuple

import h5py
import numpy as np
from model import resnet18_model


def collect_leaf_datasets(h5obj, prefix: str = "") -> Dict[str, np.ndarray]:
    """
    递归收集 HDF5 中所有叶子 dataset，返回:
    {
        "path/to/dataset": np.ndarray,
        ...
    }
    """
    items = {}
    for key in h5obj.keys():
        obj = h5obj[key]
        cur_path = f"{prefix}/{key}" if prefix else key
        if isinstance(obj, h5py.Dataset):
            items[cur_path] = obj[()]
        elif isinstance(obj, h5py.Group):
            items.update(collect_leaf_datasets(obj, cur_path))
    return items


def build_candidate_pools(all_ds: Dict[str, np.ndarray]) -> Dict[str, List[Tuple[str, np.ndarray]]]:
    """
    将源 HDF5 的所有 dataset 按可能用途分组:
    - conv_kernel
    - bn_gamma / bn_beta / bn_mean / bn_var
    - dense_kernel / dense_bias
    """
    pools = {
        "conv_kernel": [],
        "bn_gamma": [],
        "bn_beta": [],
        "bn_mean": [],
        "bn_var": [],
        "dense_kernel": [],
        "dense_bias": [],
    }

    for path, arr in all_ds.items():
        low = path.lower()

        # Conv1D kernel
        if "kernel:0" in low and "conv1d" in low:
            pools["conv_kernel"].append((path, arr))
        # Dense kernel / bias
        elif "kernel" in low and "dense" in low:
            pools["dense_kernel"].append((path, arr))
        elif "bias" in low and "dense" in low:
            pools["dense_bias"].append((path, arr))
        # BatchNorm
        elif "gamma:0" in low or low.endswith("/gamma"):
            pools["bn_gamma"].append((path, arr))
        elif "beta:0" in low or low.endswith("/beta"):
            pools["bn_beta"].append((path, arr))
        elif "moving_mean:0" in low or "moving_mean" in low:
            pools["bn_mean"].append((path, arr))
        elif "moving_variance:0" in low or "moving_variance" in low:
            pools["bn_var"].append((path, arr))

    return pools


def target_kind_from_name(name: str) -> str:
    """
    根据 PyTorch state_dict 的 key 判断需要从哪类源张量里找。
    """
    if name.endswith("conv1.weight") or name.endswith("conv2.weight") or name.endswith("skip_conv.weight"):
        return "conv_kernel"
    if name == "conv1.weight":
        return "conv_kernel"

    if name.endswith("bn1.weight") or name == "bn1.weight":
        return "bn_gamma"
    if name.endswith("bn1.bias") or name == "bn1.bias":
        return "bn_beta"
    if name.endswith("bn1.running_mean") or name == "bn1.running_mean":
        return "bn_mean"
    if name.endswith("bn1.running_var") or name == "bn1.running_var":
        return "bn_var"

    if name.endswith(".weight") and ("fc" in name):
        return "dense_kernel"
    if name.endswith(".bias") and ("fc" in name):
        return "dense_bias"

    return ""


def transform_source_for_target(kind: str, src: np.ndarray) -> np.ndarray:
    """
    将 Keras 权重转为 PyTorch 需要的形状:
    - Conv1D: (kernel, in, out) -> (out, in, kernel)
    - Dense:  (in, out) -> (out, in)
    - BN:     原样
    """
    if kind == "conv_kernel":
        if src.ndim != 3:
            raise ValueError(f"conv_kernel 期望 3 维，实际 {src.shape}")
        return np.transpose(src, (2, 1, 0))

    if kind == "dense_kernel":
        if src.ndim != 2:
            raise ValueError(f"dense_kernel 期望 2 维，实际 {src.shape}")
        return src.T

    return src


def choose_best_match(
    target_name: str,
    target_shape: Tuple[int, ...],
    kind: str,
    pools: Dict[str, List[Tuple[str, np.ndarray]]],
    used_paths: set,
) -> Tuple[str, np.ndarray]:
    """
    在对应池中寻找与 target_shape 精确匹配的唯一候选。
    优先原则:
    1. shape 完全匹配
    2. 路径字符串中若带有 block 号/层名提示，则优先
    3. 若仍有多个，按路径排序取第一个，并打印提醒
    """
    candidates = []

    for path, raw in pools.get(kind, []):
        if path in used_paths:
            continue
        try:
            converted = transform_source_for_target(kind, raw)
        except Exception:
            continue
        if tuple(converted.shape) == tuple(target_shape):
            score = 0

            # 简单加一点“语义优先级”
            low_name = target_name.lower()
            low_path = path.lower()

            for token in ["res_unit1", "res_unit2", "res_unit3", "res_unit4"]:
                if token in low_name and token in low_path:
                    score += 10

            if "skip_conv" in low_name and converted.ndim == 3 and converted.shape[-1] == 1:
                score += 5
            if ("conv1.weight" in low_name or "conv2.weight" in low_name) and converted.ndim == 3 and converted.shape[-1] == 16:
                score += 5

            candidates.append((score, path, converted))

    if not candidates:
        return "", None

    candidates.sort(key=lambda x: (-x[0], x[1]))

    if len(candidates) > 1:
        print(f"⚠️ {target_name} 存在多个同 shape 候选，已选择分数最高者:")
        for score, path, arr in candidates[:5]:
            print(f"   score={score:2d} | {path} | shape={arr.shape}")

    _, best_path, best_arr = candidates[0]
    return best_path, best_arr


def keras_hdf5_to_pytorch_shape_safe(
    keras_hdf5_path: str,
    pytorch_hdf5_path: str,
    num_classes: int = 6,
):
    """
    更稳健的 Keras HDF5 -> PyTorch HDF5 转换:
    1. 不按固定层编号映射
    2. 只按 shape 严格匹配
    3. 每个源张量只使用一次
    4. 头部不匹配时自动跳过
    """
    if not os.path.exists(keras_hdf5_path):
        raise FileNotFoundError(f"Keras HDF5 不存在: {keras_hdf5_path}")

    print(f"读取 Keras HDF5: {keras_hdf5_path}")
    model = resnet18_model(num_classes=num_classes)
    state_dict = model.state_dict()

    with h5py.File(keras_hdf5_path, "r") as f:
        all_ds = collect_leaf_datasets(f)

    print(f"共发现 {len(all_ds)} 个叶子 dataset")
    pools = build_candidate_pools(all_ds)

    print("源张量池统计:")
    for k, v in pools.items():
        print(f"  {k}: {len(v)}")

    used_paths = set()
    saved_count = 0
    missing = []
    ambiguous = []

    with h5py.File(pytorch_hdf5_path, "w") as hf:
        for name, tensor in state_dict.items():
            # 这些张量不需要从 keras 权重里加载
            if name.endswith("num_batches_tracked"):
                print(f"ℹ️ 跳过 buffer: {name}")
                continue

            kind = target_kind_from_name(name)
            if not kind:
                print(f"ℹ️ 未处理类型: {name}")
                continue

            src_path, data = choose_best_match(
                target_name=name,
                target_shape=tuple(tensor.shape),
                kind=kind,
                pools=pools,
                used_paths=used_paths,
            )

            if data is None:
                missing.append((name, tuple(tensor.shape), kind))
                print(f"✗ 未匹配: {name} | target_shape={tuple(tensor.shape)} | kind={kind}")
                continue

            hf.create_dataset(name, data=data.astype(np.float32))
            used_paths.add(src_path)
            saved_count += 1
            print(f"✓ 保存: {name} <= {src_path} | shape={data.shape}")

    print("\n" + "=" * 80)
    print(f"✅ 转换完成: {pytorch_hdf5_path}")
    print(f"✅ 成功保存张量数: {saved_count}")
    print(f"⚠️ 未匹配张量数: {len(missing)}")

    if missing:
        print("\n未匹配张量列表:")
        for name, shape, kind in missing:
            print(f"  - {name} | shape={shape} | kind={kind}")

    print(f"\n已使用源张量数: {len(used_paths)}")
    print("建议下一步：")
    print("1) 用新的 model_converted.h5 再跑一次 load_hdf5_weights")
    print("2) 理想情况是 backbone 大部分层都能加载成功")
    print("3) 若仍只有少量层成功，则说明源 Keras 架构与当前 PyTorch backbone 并不同构")


if __name__ == "__main__":
    keras_hdf5_path = r"F:\ai-ecg\core-ref\automatic-ecg-diagnosis\model.hdf5"
    pytorch_hdf5_path = r"F:\ai-ecg\core-ref\automatic-ecg-diagnosis\model_converted.h5"

    keras_hdf5_to_pytorch_shape_safe(
        keras_hdf5_path=keras_hdf5_path,
        pytorch_hdf5_path=pytorch_hdf5_path,
        num_classes=6,
    )
