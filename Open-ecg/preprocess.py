import h5py
import pandas as pd
import numpy as np
from scipy import signal
from tqdm import tqdm
import os
import argparse


class CODE15Preprocessor:
    def __init__(self, target_fs=400, target_length=4096):
        self.target_fs = target_fs
        self.target_length = target_length

    def resample_ecg(self, ecg_data, original_fs):
        if original_fs == self.target_fs:
            return ecg_data
        num_leads = ecg_data.shape[1] if ecg_data.ndim > 1 else 1
        num_samples = ecg_data.shape[0]
        duration = num_samples / original_fs
        new_num_samples = int(duration * self.target_fs)

        if ecg_data.ndim == 1:
            return signal.resample(ecg_data, new_num_samples)

        resampled = np.zeros((new_num_samples, num_leads))
        for lead in range(num_leads):
            resampled[:, lead] = signal.resample(ecg_data[:, lead], new_num_samples)
        return resampled

    def normalize_ecg(self, ecg_data, method='zscore'):
        if method == 'zscore':
            mean = np.mean(ecg_data, axis=0)
            std = np.std(ecg_data, axis=0)
            std[std == 0] = 1e-10
            return (ecg_data - mean) / std
        elif method == 'minmax':
            min_val = np.min(ecg_data, axis=0)
            max_val = np.max(ecg_data, axis=0)
            range_val = max_val - min_val
            range_val[range_val == 0] = 1e-10
            return (ecg_data - min_val) / range_val
        return ecg_data

    def crop_or_pad(self, ecg_data):
        n_samples = ecg_data.shape[0]
        if n_samples >= self.target_length:
            return ecg_data[-self.target_length:, :]
        else:
            result = np.zeros((self.target_length, ecg_data.shape[1]))
            result[-n_samples:, :] = ecg_data
            return result

    def process_single_record(self, ecg_raw, original_fs=400):
        if ecg_raw.shape[0] == 12 and ecg_raw.shape[1] != 12:
            ecg_raw = ecg_raw.T

        ecg = self.resample_ecg(ecg_raw, original_fs)
        ecg = self.normalize_ecg(ecg, method='zscore')
        ecg = self.crop_or_pad(ecg)
        return ecg


def prepare_code15_data(
        exams_csv_path,
        hdf5_dir,
        output_data_dir,
        target_fs=400,
        max_samples=None,
        balance=True,
        include_normal=False,
        seed=42
):
    """
    CODE-15 预处理 + 按类尽量均衡采样

    参数：
    - max_samples: 目标总样本数
    - balance=True: 按类均衡
    - include_normal=False: 是否把 Normal(6) 也纳入均衡采样
    """
    os.makedirs(output_data_dir, exist_ok=True)

    print(f"Loading {exams_csv_path}...")
    df = pd.read_csv(exams_csv_path)

    exam_id_to_idx = {str(eid): idx for idx, eid in enumerate(df['exam_id'].values)}
    preprocessor = CODE15Preprocessor(target_fs=target_fs)
    label_cols = ['1dAVb', 'RBBB', 'LBBB', 'SB', 'AF', 'ST']

    rng = np.random.RandomState(seed)

    # 目标类别
    target_labels = [0, 1, 2, 3, 4, 5]
    if include_normal:
        target_labels.append(6)

    # 按类配额
    class_quota = None
    if balance and max_samples is not None:
        base = max_samples // len(target_labels)
        remainder = max_samples % len(target_labels)
        class_quota = {lab: base for lab in target_labels}
        for lab in target_labels[:remainder]:
            class_quota[lab] += 1

    print(f"Target labels: {target_labels}")
    if class_quota is not None:
        print(f"Class quota: {class_quota}")

    processed_count = 0
    skipped_count = 0
    class_counts = {lab: 0 for lab in target_labels}
    valid_records = []

    normal_count = 0
    single_label_count = 0
    multi_label_skipped = 0

    hdf5_files = sorted([f for f in os.listdir(hdf5_dir) if f.endswith('.hdf5') and 'exams_part' in f])
    print(f"Found {len(hdf5_files)} HDF5 files: {hdf5_files}")

    stop_flag = False

    for hdf5_file in hdf5_files:
        if stop_flag:
            break

        hdf5_path = os.path.join(hdf5_dir, hdf5_file)
        print(f"\nProcessing {hdf5_file}...")

        with h5py.File(hdf5_path, 'r') as f:
            tracings = f['tracings'][:]

            if 'exam_id' in f:
                exam_ids = f['exam_id'][:]
                if isinstance(exam_ids[0], bytes):
                    exam_ids = [eid.decode('utf-8') for eid in exam_ids]
                else:
                    exam_ids = [str(int(eid)) for eid in exam_ids]
            else:
                exam_ids = [f"{hdf5_file}_{i}" for i in range(len(tracings))]

            indices = np.arange(len(tracings))
            rng.shuffle(indices)  # 打乱当前文件内部顺序，减少顺序偏差

            for i in tqdm(indices, desc=hdf5_file):
                exam_id = exam_ids[i]

                if exam_id not in exam_id_to_idx:
                    continue

                row_idx = exam_id_to_idx[exam_id]
                row = df.iloc[row_idx]

                abnormality_flags = [row[col] == True or row[col] == 1 for col in label_cols]
                true_labels = [j for j, flag in enumerate(abnormality_flags) if flag]

                # 只保留 normal 或单标签异常；多标签直接过滤
                if len(true_labels) == 0:
                    if include_normal:
                        label = 6
                        normal_count += 1
                    else:
                        skipped_count += 1
                        continue

                elif len(true_labels) == 1:
                    label = true_labels[0]
                    single_label_count += 1

                else:
                    multi_label_skipped += 1
                    skipped_count += 1
                    continue

                if label not in target_labels:
                    skipped_count += 1
                    continue

                if class_quota is not None and class_counts[label] >= class_quota[label]:
                    continue

                try:
                    processed = preprocessor.process_single_record(tracings[i], target_fs)
                    save_path = os.path.join(output_data_dir, f"{exam_id}.csv")
                    pd.DataFrame(processed).to_csv(save_path, index=False, header=False)

                    valid_records.append({
                        'Recording': f"{exam_id}.csv",
                        'label': label,
                        'exam_id': exam_id
                    })

                    class_counts[label] += 1
                    processed_count += 1

                    if class_quota is not None:
                        if all(class_counts[lab] >= class_quota[lab] for lab in target_labels):
                            stop_flag = True
                            break
                    elif max_samples is not None and processed_count >= max_samples:
                        stop_flag = True
                        break

                except Exception as e:
                    print(f"Error processing {exam_id}: {e}")
                    continue

    if valid_records:
        label_df = pd.DataFrame(valid_records)

        output_csv = os.path.join(output_data_dir, 'labels.csv')
        label_df[['Recording', 'label']].to_csv(output_csv, index=False)

        print(f"\n预处理完成！")
        print(f"- 有效样本数: {processed_count}")
        print(f"- 跳过样本数: {skipped_count}")
        print(f"- 各类别分布:")
        print(label_df['label'].value_counts().sort_index())
        print(f"- 标签文件已保存至: {output_csv}")
    else:
        print("警告：没有生成有效样本！")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CODE-15 数据预处理")
    parser.add_argument('--exams_csv', type=str, required=True,
                        help='exams.csv 文件路径')
    parser.add_argument('--hdf5_dir', type=str, required=True,
                        help='HDF5 文件所在目录')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='输出 CSV 目录')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='最大处理样本数（用于测试，默认处理全部）')
    parser.add_argument('--balance', action='store_true', help='是否按类均衡采样')
    parser.add_argument('--include_normal', action='store_true', help='是否将Normal(6)也纳入采样')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')

    args = parser.parse_args()

    prepare_code15_data(
        args.exams_csv,
        args.hdf5_dir,
        args.output_dir,
        max_samples=args.max_samples,
        balance=args.balance,
        include_normal=args.include_normal,
        seed=args.seed
    )