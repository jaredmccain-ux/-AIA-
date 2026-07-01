#!/usr/bin/env python
"""
Convert DEAP .dat (pickle) files to the .mat DE feature format
expected by the original TDMNN untitled_deap.py.

Pipeline:
  Step 1 (DEAP_1D.py logic):  .dat -> 1D DE features + baseline DE
  Step 2 (DEAP_1D_3D.py logic): baseline subtraction + standardization + 2D topomap

Output: one .mat per subject with keys: data, valence_labels, arousal_labels
"""

import math
import os
import pickle
import numpy as np
import scipy.io as sio
from scipy.signal import butter, lfilter
from sklearn import preprocessing


def butter_bandpass_filter(data, lowcut, highcut, fs, order=3):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype="band")
    return lfilter(b, a, data)


def compute_DE(signal):
    variance = np.var(signal, ddof=1)
    return math.log(2 * math.pi * math.e * variance) / 2


def data_1Dto2D(data, Y=8, X=9):
    """Map 32-channel 1D vector to 8x9 2D topomap (DEAP channel layout)."""
    data_2D = np.zeros([Y, X])
    data_2D[0] = (0, 0, data[1], data[0], 0, data[16], data[17], 0, 0)
    data_2D[1] = (data[3], 0, data[2], 0, data[18], 0, data[19], 0, data[20])
    data_2D[2] = (0, data[4], 0, data[5], 0, data[22], 0, data[21], 0)
    data_2D[3] = (data[7], 0, data[6], 0, data[23], 0, data[24], 0, data[25])
    data_2D[4] = (0, data[8], 0, data[9], 0, data[27], 0, data[26], 0)
    data_2D[5] = (data[11], 0, data[10], 0, data[15], 0, data[28], 0, data[29])
    data_2D[6] = (0, 0, 0, data[12], 0, data[30], 0, 0, 0)
    data_2D[7] = (0, 0, 0, data[13], data[14], data[31], 0, 0, 0)
    return data_2D


def process_subject(dat_path):
    """Full pipeline for one subject: .dat -> DE features -> 3D topomaps."""
    frequency = 128
    start_index = 3 * frequency  # 384 samples baseline

    with open(dat_path, "rb") as f:
        obj = pickle.load(f, encoding="latin1")
    raw_data = np.asarray(obj["data"], dtype=np.float64)
    labels = np.asarray(obj["labels"], dtype=np.float64)

    # --- Step 1: Extract 1D DE features (DEAP_1D.py logic) ---
    decomposed_de = np.empty([0, 4, 120])
    base_DE = np.empty([0, 128])

    for trial in range(40):
        temp_base_DE = np.empty([0])
        temp_base_theta_DE = np.empty([0])
        temp_base_alpha_DE = np.empty([0])
        temp_base_beta_DE = np.empty([0])
        temp_base_gamma_DE = np.empty([0])
        temp_de = np.empty([0, 120])

        for channel in range(32):
            trial_signal = raw_data[trial, channel, start_index:]
            base_signal = raw_data[trial, channel, :start_index]

            base_theta = butter_bandpass_filter(base_signal, 4, 8, frequency, order=3)
            base_alpha = butter_bandpass_filter(base_signal, 8, 14, frequency, order=3)
            base_beta = butter_bandpass_filter(base_signal, 14, 31, frequency, order=3)
            base_gamma = butter_bandpass_filter(base_signal, 31, 45, frequency, order=3)

            # NOTE: Original code has a bug where base_alpha/beta/gamma DE
            # uses base_theta for some windows. We replicate it exactly.
            base_theta_DE = (
                compute_DE(base_theta[:64]) + compute_DE(base_theta[64:128])
                + compute_DE(base_theta[128:192]) + compute_DE(base_theta[192:256])
                + compute_DE(base_theta[256:320]) + compute_DE(base_theta[320:])
            ) / 6
            base_alpha_DE = (
                compute_DE(base_alpha[:64]) + compute_DE(base_alpha[64:128])
                + compute_DE(base_alpha[128:192]) + compute_DE(base_theta[192:256])
                + compute_DE(base_theta[256:320]) + compute_DE(base_theta[320:])
            ) / 6
            base_beta_DE = (
                compute_DE(base_beta[:64]) + compute_DE(base_beta[64:128])
                + compute_DE(base_beta[128:192]) + compute_DE(base_theta[192:256])
                + compute_DE(base_theta[256:320]) + compute_DE(base_theta[320:])
            ) / 6
            base_gamma_DE = (
                compute_DE(base_gamma[:64]) + compute_DE(base_gamma[64:128])
                + compute_DE(base_gamma[128:192]) + compute_DE(base_theta[192:256])
                + compute_DE(base_theta[256:320]) + compute_DE(base_theta[320:])
            ) / 6

            temp_base_theta_DE = np.append(temp_base_theta_DE, base_theta_DE)
            temp_base_alpha_DE = np.append(temp_base_alpha_DE, base_alpha_DE)
            temp_base_beta_DE = np.append(temp_base_beta_DE, base_beta_DE)
            temp_base_gamma_DE = np.append(temp_base_gamma_DE, base_gamma_DE)

            theta = butter_bandpass_filter(trial_signal, 4, 8, frequency, order=3)
            alpha = butter_bandpass_filter(trial_signal, 8, 14, frequency, order=3)
            beta = butter_bandpass_filter(trial_signal, 14, 31, frequency, order=3)
            gamma = butter_bandpass_filter(trial_signal, 31, 45, frequency, order=3)

            DE_theta = np.array([compute_DE(theta[i * 64:(i + 1) * 64]) for i in range(120)])
            DE_alpha = np.array([compute_DE(alpha[i * 64:(i + 1) * 64]) for i in range(120)])
            DE_beta = np.array([compute_DE(beta[i * 64:(i + 1) * 64]) for i in range(120)])
            DE_gamma = np.array([compute_DE(gamma[i * 64:(i + 1) * 64]) for i in range(120)])

            temp_de = np.vstack([temp_de, DE_theta, DE_alpha, DE_beta, DE_gamma])

        temp_trial_de = temp_de.reshape(-1, 4, 120)
        decomposed_de = np.vstack([decomposed_de, temp_trial_de])

        temp_base_DE_full = np.concatenate([
            temp_base_theta_DE, temp_base_alpha_DE,
            temp_base_beta_DE, temp_base_gamma_DE
        ])
        base_DE = np.vstack([base_DE, temp_base_DE_full])

    # Reshape to (4800, 128)
    trial_DE = decomposed_de.reshape(-1, 32, 4, 120).transpose([0, 3, 2, 1]).reshape(-1, 4, 32).reshape(-1, 128)

    # --- Step 2: Baseline subtraction + standardization + 2D (DEAP_1D_3D.py logic) ---
    new_dataset = np.empty([0, 128])
    for i in range(4800):
        base_index = min(i // 120, 39)
        new_record = (trial_DE[i] - base_DE[base_index]).reshape(1, 128)
        new_dataset = np.vstack([new_dataset, new_record])

    data_scaled = preprocessing.scale(new_dataset, axis=1, with_mean=True, with_std=True, copy=True)

    sub_vector_len = 32
    data_3D = np.empty([0, 8, 9])
    for vector in data_scaled:
        for band in range(4):
            data_2D_temp = data_1Dto2D(vector[band * sub_vector_len:(band + 1) * sub_vector_len])
            data_3D = np.vstack([data_3D, data_2D_temp.reshape(1, 8, 9)])
    data_3D = data_3D.reshape(-1, 4, 8, 9)

    valence_labels = (labels[:, 0] > 5).astype(float)
    arousal_labels = (labels[:, 1] > 5).astype(float)
    final_valence = np.repeat(valence_labels, 120)
    final_arousal = np.repeat(arousal_labels, 120)

    return data_3D, final_valence, final_arousal


if __name__ == "__main__":
    dat_dir = "/root/inpainting-work/interactive/new/data_preprocessed_custom"
    out_dir = "/root/inpainting-work/interactive/TDMNN/deap_mat"
    os.makedirs(out_dir, exist_ok=True)

    for sid in range(1, 33):
        dat_path = os.path.join(dat_dir, f"s{sid:02d}.dat")
        if not os.path.exists(dat_path):
            print(f"  s{sid:02d}: SKIPPED (file not found)")
            continue
        print(f"Processing s{sid:02d}...", end=" ", flush=True)
        data_3D, valence_labels, arousal_labels = process_subject(dat_path)
        out_path = os.path.join(out_dir, f"DE_s{sid:02d}.mat")
        sio.savemat(out_path, {
            "data": data_3D,
            "valence_labels": valence_labels,
            "arousal_labels": arousal_labels,
        })
        print(f"done -> {data_3D.shape}")
