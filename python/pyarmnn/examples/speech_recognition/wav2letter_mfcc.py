# Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
# SPDX-License-Identifier: MIT

import numpy as np
import os
import sys

script_dir = os.path.dirname(__file__)
sys.path.insert(1, os.path.join(script_dir, '..', 'common'))

from mfcc import MFCC, AudioPreprocessor


class Wav2LetterMFCC(MFCC):
    """Extends base MFCC class to provide Wav2Letter-specific MFCC requirements."""

    def __init__(self, mfcc_params):
        super().__init__(mfcc_params)

    def spectrum_calc(self, audio_data):
        return np.abs(np.fft.rfft(np.hanning(self.mfcc_params.frame_len + 1)[0:self.mfcc_params.frame_len] * audio_data,
                                  self.mfcc_params.n_fft)) ** 2

    def log_mel(self, mel_energy):
        mel_energy += 1e-10
        log_mel_energy = 10.0 * np.log10(mel_energy)
        top_db = 80.0
        return np.maximum(log_mel_energy, log_mel_energy.max() - top_db)

    def create_dct_matrix(self, num_fbank_bins, num_mfcc_feats):
        """
        Creates the Discrete Cosine Transform matrix to be used in the compute function.

        Args:
            num_fbank_bins: The number of filter bank bins
            num_mfcc_feats: the number of MFCC features

        Returns:
            the DCT matrix
        """
        dct_m = np.zeros(num_fbank_bins * num_mfcc_feats)
        for k in range(num_mfcc_feats):
            for n in range(num_fbank_bins):
                if k == 0:
                    dct_m[(k * num_fbank_bins) + n] = 2 * np.sqrt(1 / (4 * num_fbank_bins)) * np.cos(
                        (np.pi / num_fbank_bins) * (n + 0.5) * k)
                else:
                    dct_m[(k * num_fbank_bins) + n] = 2 * np.sqrt(1 / (2 * num_fbank_bins)) * np.cos(
                        (np.pi / num_fbank_bins) * (n + 0.5) * k)

        dct_m = np.reshape(dct_m, [self.mfcc_params.num_mfcc_feats, self.mfcc_params.num_fbank_bins])
        return dct_m

    def mel_norm(self, weight, right_mel, left_mel):
        """Over-riding parent class with ASR specific weight normalisation."""
        enorm = 2.0 / (self.inv_mel_scale(right_mel, False) - self.inv_mel_scale(left_mel, False))
        return weight * enorm


class W2LAudioPreprocessor(AudioPreprocessor):

    def __init__(self, mfcc, model_input_size, stride):
        self.model_input_size = model_input_size
        self.stride = stride

        super().__init__(self, model_input_size, stride)
        # Savitzky - Golay differential filters
        self.savgol_order1_coeffs = np.array([6.66666667e-02, 5.00000000e-02, 3.33333333e-02,
                                              1.66666667e-02, -3.46944695e-18, -1.66666667e-02,
                                              -3.33333333e-02, -5.00000000e-02, -6.66666667e-02])

        self.savgol_order2_coeffs = np.array([0.06060606, 0.01515152, -0.01731602,
                                              -0.03679654, -0.04329004, -0.03679654,
                                              -0.01731602, 0.01515152, 0.06060606])
        self._mfcc_calc = mfcc

    def mfcc_delta_calc(self, features):
        """Over-riding parent class with ASR specific MFCC derivative features"""
        mfcc_delta_np = np.zeros_like(features)
        mfcc_delta2_np = np.zeros_like(features)

        for i in range(features.shape[1]):
            idelta = np.convolve(features[:, i], self.savgol_order1_coeffs, 'same')
            mfcc_delta_np[:, i] = idelta
            ideltadelta = np.convolve(features[:, i], self.savgol_order2_coeffs, 'same')
            mfcc_delta2_np[:, i] = ideltadelta

        features = np.concatenate((self._normalize(features), self._normalize(mfcc_delta_np),
                                   self._normalize(mfcc_delta2_np)), axis=1)

        return features
