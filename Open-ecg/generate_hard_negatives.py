import numpy as np
import pandas as pd
import neurokit2 as nk
from scipy import signal
from scipy.interpolate import interp1d
import os
from tqdm import tqdm
import argparse


class HardNegativeGenerator:
    """
    基于临床知识生成硬负样本（对应论文 Table 1）
    为CODE-15的6类异常生成对应的硬负样本
    """

    def __init__(self, sampling_rate=400, beta=3.0, theta=1.3):
        self.fs = sampling_rate
        self.beta = beta  # 诊断区域变换强度
        self.theta = theta  # 非诊断区域轻微变换

        # 与 preprocess.py 保持一致
        # 0: 1dAVb, 1: RBBB, 2: LBBB, 3: SB, 4: AF, 5: ST, 6: Normal
        self.transform_map = {
            0: self.transform_1dAVb,
            1: self.transform_RBBB,
            2: self.transform_LBBB,
            3: self.transform_SB,
            4: self.transform_AF,
            5: self.transform_ST,
            6: self.transform_normal,
        }

    def delineate_ecg(self, ecg_signal):
        """使用neurokit2进行ECG波形分割"""
        try:
            if ecg_signal.ndim > 1:
                lead_signal = ecg_signal[:, 0]  # 使用Lead I
            else:
                lead_signal = ecg_signal

            # 清理和检测R峰
            signals, info = nk.ecg_process(lead_signal, sampling_rate=self.fs)
            r_peaks = info['ECG_R_Peaks']

            if len(r_peaks) < 2:
                return self._fallback_segmentation(lead_signal)

            # 获取P波、T波位置
            _, waves = nk.ecg_delineate(lead_signal, r_peaks, sampling_rate=self.fs, method="peak")

            r_peak = r_peaks[0]
            p_onset = waves['ECG_P_Onsets'][0] if not pd.isna(waves['ECG_P_Onsets'][0]) else r_peak - int(
                0.12 * self.fs)
            p_offset = waves['ECG_P_Offsets'][0] if not pd.isna(waves['ECG_P_Offsets'][0]) else r_peak - int(
                0.02 * self.fs)
            t_offset = waves['ECG_T_Offsets'][0] if not pd.isna(waves['ECG_T_Offsets'][0]) else r_peak + int(
                0.3 * self.fs)
            q_peak = waves['ECG_Q_Peaks'][0] if not pd.isna(waves['ECG_Q_Peaks'][0]) else r_peak - 10
            s_peak = waves['ECG_S_Peaks'][0] if not pd.isna(waves['ECG_S_Peaks'][0]) else r_peak + 10

            return {
                'p_wave': (max(0, int(p_onset)), max(0, int(p_offset))),
                'pr_interval': (max(0, int(p_onset)), r_peak),
                'qrs': (int(q_peak), int(s_peak)),
                'st_segment': (int(s_peak),
                               int(waves['ECG_T_Onsets'][0]) if not pd.isna(waves['ECG_T_Onsets'][0]) else r_peak + 40),
                't_wave': (int(waves['ECG_T_Onsets'][0]) if not pd.isna(waves['ECG_T_Onsets'][0]) else r_peak + 40,
                           int(t_offset)),
            }
        except Exception as e:
            return self._fallback_segmentation(ecg_signal)

    def _fallback_segmentation(self, signal):
        """当neurokit2失败时的备用方案"""
        if len(signal) < self.fs:
            return None
        r_peak = self.fs // 2  # 假设R峰在中间
        return {
            'p_wave': (r_peak - int(0.2 * self.fs), r_peak - int(0.05 * self.fs)),
            'pr_interval': (r_peak - int(0.2 * self.fs), r_peak),
            'qrs': (r_peak - 10, r_peak + 10),
            'st_segment': (r_peak + 10, r_peak + 50),
            't_wave': (r_peak + 50, r_peak + 150),
        }

    def stretch_segment(self, signal, start, end, factor):
        """拉伸信号片段"""
        if start >= end or start < 0 or end > len(signal):
            return signal
        segment = signal[start:end]
        old_x = np.arange(len(segment))
        new_length = int(len(segment) * factor)
        new_x = np.linspace(0, len(segment) - 1, new_length)
        f = interp1d(old_x, segment, kind='linear', fill_value='extrapolate')
        stretched = f(new_x)
        # 截断或填充回原长度
        if len(stretched) > len(segment):
            signal[start:end] = stretched[:len(segment)]
        else:
            signal[start:start + len(stretched)] = stretched
        return signal

    def scale_segment(self, signal, start, end, factor):
        """缩放信号幅度"""
        if start >= end or start < 0 or end > len(signal):
            return signal
        signal[start:end] *= factor
        return signal

    def mask_segment(self, signal, start, end):
        """掩码信号（置为基线）"""
        if start >= end or start < 0 or end > len(signal):
            return signal
        baseline = np.mean(signal[:100]) if len(signal) > 100 else 0
        signal[start:end] = baseline
        return signal

    # ===== 各类别的具体变换策略（对应Table 1）=====
    def transform_normal(self, ecg):
        """Normal: 整体放大"""
        return ecg * self.beta

    def transform_1dAVb(self, ecg):
        """1dAVb: 拉伸PR间期（模拟更长PR）"""
        segs = self.delineate_ecg(ecg)
        if segs is None: return ecg * self.beta

        ecg_new = ecg.copy()
        pr_start, pr_end = segs['pr_interval']
        # 仅对PR间期拉伸
        ecg_new = self.stretch_segment(ecg_new, pr_start, pr_end, self.beta)
        # 其他区域轻微缩放
        mask = np.ones(len(ecg_new), dtype=bool)
        mask[pr_start:pr_end] = False
        ecg_new[mask] *= self.theta
        return ecg_new

    def transform_RBBB(self, ecg):
        """RBBB: 压缩QRS并放大"""
        segs = self.delineate_ecg(ecg)
        if segs is None: return ecg * self.beta

        ecg_new = ecg.copy()
        qrs_start, qrs_end = segs['qrs']
        # 先压缩时间
        ecg_new = self.stretch_segment(ecg_new, qrs_start, qrs_end, 1 / self.beta)
        # 再放大幅度
        ecg_new = self.scale_segment(ecg_new, qrs_start, qrs_end, self.beta)
        return ecg_new

    def transform_LBBB(self, ecg):
        """LBBB: 放大QRS"""
        segs = self.delineate_ecg(ecg)
        if segs is None: return ecg * self.beta

        ecg_new = ecg.copy()
        qrs_start, qrs_end = segs['qrs']
        ecg_new = self.stretch_segment(ecg_new, qrs_start, qrs_end, self.beta)
        return ecg_new

    def transform_SB(self, ecg):
        """SB: 拉伸PR间期"""
        return self.transform_1dAVb(ecg)  # 类似策略

    def transform_AF(self, ecg):
        """AF: 放大PR间期（与AF的'无P波'特征相反）"""
        segs = self.delineate_ecg(ecg)
        if segs is None: return ecg * self.beta

        ecg_new = ecg.copy()
        pr_start, pr_end = segs['pr_interval']
        # 放大PR间期幅度（模拟出现明显P波）
        ecg_new = self.scale_segment(ecg_new, pr_start, pr_end, self.beta)
        return ecg_new

    def transform_ST(self, ecg):
        """ST: 压缩PR间期（心率快，PR短）"""
        segs = self.delineate_ecg(ecg)
        if segs is None: return ecg * self.beta

        ecg_new = ecg.copy()
        pr_start, pr_end = segs['pr_interval']
        # 压缩PR间期
        ecg_new = self.stretch_segment(ecg_new, pr_start, pr_end, 1 / self.beta)
        return ecg_new

    def generate_for_file(self, input_path, output_path, label):
        try:
            df = pd.read_csv(input_path, header=None)
            ecg = df.values.astype(np.float32)

            ecg_hardneg = np.zeros_like(ecg)
            transform_fn = self.transform_map.get(label, self.transform_normal)

            for lead in range(ecg.shape[1]):
                ecg_hardneg[:, lead] = transform_fn(ecg[:, lead])

            pd.DataFrame(ecg_hardneg).to_csv(output_path, index=False, header=False)
            return True
        except Exception as e:
            print(f"Error processing {input_path}: {e}")
            return False

    def batch_generate(self, label_csv_path, data_dir, output_dir):
        """
        批量生成硬负样本
        生成格式: hardneg_{original_name}.csv
        """
        os.makedirs(output_dir, exist_ok=True)
        df = pd.read_csv(label_csv_path)

        print(f"Generating hard negatives for {len(df)} files...")
        success_count = 0

        for idx, row in tqdm(df.iterrows(), total=len(df)):
            filename = row['Recording']  # 假设列名为Recording
            label = int(row['label'])

            input_path = os.path.join(data_dir, filename)
            output_filename = f"hardneg_{filename}"
            output_path = os.path.join(output_dir, output_filename)

            if os.path.exists(output_path):
                success_count += 1
                continue

            if os.path.exists(input_path):
                if self.generate_for_file(input_path, output_path, label):
                    success_count += 1

        print(f"Successfully generated {success_count} hard negative files in {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='存放原始CSV的目录')
    parser.add_argument('--label_csv', type=str, required=True, help='标签CSV路径')
    parser.add_argument('--output_dir', type=str, default=None, help='输出目录，默认同data_dir')
    parser.add_argument('--fs', type=int, default=400, help='采样率')
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = args.data_dir

    generator = HardNegativeGenerator(sampling_rate=args.fs)
    generator.batch_generate(args.label_csv, args.data_dir, args.output_dir)