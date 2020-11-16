# Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
# SPDX-License-Identifier: MIT

import numpy as np

from context import preprocess

test_wav = [
    -3,0,1,-1,2,3,-2,2,
    1,-2,0,3,-1,8,3,2,
    -1,-1,2,7,3,5,6,6,
    6,12,5,6,3,3,5,4,
    4,6,7,7,7,3,7,2,
    8,4,4,2,-4,-1,-1,-4,
    2,1,-1,-4,0,-7,-6,-2,
    -5,1,-5,-1,-7,-3,-3,-7,
    0,-3,3,-5,0,1,-2,-2,
    -3,-3,-7,-3,-2,-6,-5,-8,
    -2,-8,4,-9,-4,-9,-5,-5,
    -3,-9,-3,-9,-1,-7,-4,1,
    -3,2,-8,-4,-4,-5,1,-3,
    -1,0,-1,-2,-3,-2,-4,-1,
    1,-1,3,0,3,2,0,0,
    0,-3,1,1,0,8,3,4,
    1,5,6,4,7,3,3,0,
    3,6,7,6,4,5,9,9,
    5,5,8,1,6,9,6,6,
    7,1,8,1,5,0,5,5,
    0,3,2,7,2,-3,3,0,
    3,0,0,0,2,0,-1,-1,
    -2,-3,-8,0,1,0,-3,-3,
    -3,-2,-3,-3,-4,-6,-2,-8,
    -9,-4,-1,-5,-3,-3,-4,-3,
    -6,3,0,-1,-2,-9,-4,-2,
    2,-1,3,-5,-5,-2,0,-2,
    0,-1,-3,1,-2,9,4,5,
    2,2,1,0,-6,-2,0,0,
    0,-1,4,-4,3,-7,-1,5,
    -6,-1,-5,4,3,9,-2,1,
    3,0,0,-2,1,2,1,1,
    0,3,2,-1,3,-3,7,0,
    0,3,2,2,-2,3,-2,2,
    -3,4,-1,-1,-5,-1,-3,-2,
    1,-1,3,2,4,1,2,-2,
    0,2,7,0,8,-3,6,-3,
    6,1,2,-3,-1,-1,-1,1,
    -2,2,1,2,0,-2,3,-2,
    3,-2,1,0,-3,-1,-2,-4,
    -6,-5,-8,-1,-4,0,-3,-1,
    -1,-1,0,-2,-3,-7,-1,0,
    1,5,0,5,1,1,-3,0,
    -6,3,-8,4,-8,6,-6,1,
    -6,-2,-5,-6,0,-5,4,-1,
    4,-2,1,2,1,0,-2,0,
    0,2,-2,2,-5,2,0,-2,
    1,-2,0,5,1,0,1,5,
    0,8,3,2,2,0,5,-2,
    3,1,0,1,0,-2,-1,-3,
    1,-1,3,0,3,0,-2,-1,
    -4,-4,-4,-1,-4,-4,-3,-6,
    -3,-7,-3,-1,-2,0,-5,-4,
    -7,-3,-2,-2,1,2,2,8,
    5,4,2,4,3,5,0,3,
    3,6,4,2,2,-2,4,-2,
    3,3,2,1,1,4,-5,2,
    -3,0,-1,1,-2,2,5,1,
    4,2,3,1,-1,1,0,6,
    0,-2,-1,1,-1,2,-5,-1,
    -5,-1,-6,-3,-3,2,4,0,
    -1,-5,3,-4,-1,-3,-4,1,
    -4,1,-1,-1,0,-5,-4,-2,
    -1,-1,-3,-7,-3,-3,4,4,
]

def test_mel_scale_function_with_htk_true():
    samp_freq = 16000
    frame_len_ms = 32
    frame_len_samples = samp_freq * frame_len_ms * 0.001
    num_mfcc_feats = 13
    num_fbank_bins = 128
    mel_lo_freq = 0
    mil_hi_freq = 8000
    use_htk = False
    n_FFT = 512

    mfcc_params = preprocess.MFCCParams(samp_freq, num_fbank_bins, mel_lo_freq, mil_hi_freq, num_mfcc_feats,
                                          frame_len_samples, use_htk, n_FFT)

    mfcc_inst = preprocess.MFCC(mfcc_params)

    mel = mfcc_inst.mel_scale(16, True)

    assert np.isclose(mel, 25.470010570730597)


def test_mel_scale_function_with_htk_false():
    samp_freq = 16000
    frame_len_ms = 32
    frame_len_samples = samp_freq * frame_len_ms * 0.001
    num_mfcc_feats = 13
    num_fbank_bins = 128
    mel_lo_freq = 0
    mil_hi_freq = 8000
    use_htk = False
    n_FFT = 512

    mfcc_params = preprocess.MFCCParams(samp_freq, num_fbank_bins, mel_lo_freq, mil_hi_freq, num_mfcc_feats,
                                          frame_len_samples, use_htk, n_FFT)

    mfcc_inst = preprocess.MFCC(mfcc_params)

    mel = mfcc_inst.mel_scale(16, False)

    assert np.isclose(mel, 0.24)


def test_inverse_mel_scale_function_with_htk_true():
    samp_freq = 16000
    frame_len_ms = 32
    frame_len_samples = samp_freq * frame_len_ms * 0.001
    num_mfcc_feats = 13
    num_fbank_bins = 128
    mel_lo_freq = 0
    mil_hi_freq = 8000
    use_htk = False
    n_FFT = 512

    mfcc_params = preprocess.MFCCParams(samp_freq, num_fbank_bins, mel_lo_freq, mil_hi_freq, num_mfcc_feats,
                                          frame_len_samples, use_htk, n_FFT)

    mfcc_inst = preprocess.MFCC(mfcc_params)

    mel = mfcc_inst.inv_mel_scale(16, True)

    assert np.isclose(mel, 10.008767240008943)


def test_inverse_mel_scale_function_with_htk_false():
    samp_freq = 16000
    frame_len_ms = 32
    frame_len_samples = samp_freq * frame_len_ms * 0.001
    num_mfcc_feats = 13
    num_fbank_bins = 128
    mel_lo_freq = 0
    mil_hi_freq = 8000
    use_htk = False
    n_FFT = 512

    mfcc_params = preprocess.MFCCParams(samp_freq, num_fbank_bins, mel_lo_freq, mil_hi_freq, num_mfcc_feats,
                                          frame_len_samples, use_htk, n_FFT)

    mfcc_inst = preprocess.MFCC(mfcc_params)

    mel = mfcc_inst.inv_mel_scale(16, False)

    assert np.isclose(mel, 1071.170287494467)


def test_create_mel_filter_bank():
    samp_freq = 16000
    frame_len_ms = 32
    frame_len_samples = samp_freq * frame_len_ms * 0.001
    num_mfcc_feats = 13
    num_fbank_bins = 128
    mel_lo_freq = 0
    mil_hi_freq = 8000
    use_htk = False
    n_FFT = 512

    mfcc_params = preprocess.MFCCParams(samp_freq, num_fbank_bins, mel_lo_freq, mil_hi_freq, num_mfcc_feats,
                                          frame_len_samples, use_htk, n_FFT)

    mfcc_inst = preprocess.MFCC(mfcc_params)

    mel_filter_bank = mfcc_inst.create_mel_filter_bank()

    assert len(mel_filter_bank) == 128

    assert str(mel_filter_bank[0]) == "[0.02837754]"
    assert str(mel_filter_bank[1]) == "[0.01438901 0.01398853]"
    assert str(mel_filter_bank[2]) == "[0.02877802]"
    assert str(mel_filter_bank[3]) == "[0.04236608]"
    assert str(mel_filter_bank[4]) == "[0.00040047 0.02797707]"
    assert str(mel_filter_bank[5]) == "[0.01478948 0.01358806]"
    assert str(mel_filter_bank[50]) == "[0.03298853]"
    assert str(mel_filter_bank[100]) == "[0.00260166 0.00588759 0.00914814 0.00798015 0.00476919 0.00158245]"


def test_mfcc_compute():
    samp_freq = 16000
    frame_len_ms = 32
    frame_len_samples = samp_freq * frame_len_ms * 0.001
    num_mfcc_feats = 13
    num_fbank_bins = 128
    mel_lo_freq = 0
    mil_hi_freq = 8000
    use_htk = False
    n_FFT = 512

    audio_data = np.array(test_wav) / (2 ** 15)

    mfcc_params = preprocess.MFCCParams(samp_freq, num_fbank_bins, mel_lo_freq, mil_hi_freq, num_mfcc_feats,
                                          frame_len_samples, use_htk, n_FFT)
    mfcc_inst = preprocess.MFCC(mfcc_params)
    mfcc_feats = mfcc_inst.mfcc_compute(audio_data)

    assert np.isclose((mfcc_feats[0]), -834.9656973095651)
    assert np.isclose((mfcc_feats[1]), 21.026915475076322)
    assert np.isclose((mfcc_feats[2]), 18.628541708201688)
    assert np.isclose((mfcc_feats[3]), 7.341153529494758)
    assert np.isclose((mfcc_feats[4]), 18.907974386153214)
    assert np.isclose((mfcc_feats[5]), -5.360387487466194)
    assert np.isclose((mfcc_feats[6]), 6.523572638527085)
    assert np.isclose((mfcc_feats[7]), -11.270643644983316)
    assert np.isclose((mfcc_feats[8]), 8.375177203773777)
    assert np.isclose((mfcc_feats[9]), 12.06721844362991)
    assert np.isclose((mfcc_feats[10]), 8.30815892468875)
    assert np.isclose((mfcc_feats[11]), -13.499911910889917)
    assert np.isclose((mfcc_feats[12]), -18.176121251436165)


def test_sliding_window_for_small_num_samples():
    samp_freq = 16000
    frame_len_ms = 32
    frame_len_samples = samp_freq * frame_len_ms * 0.001
    num_mfcc_feats = 13
    mode_input_size = 9
    stride = 160
    num_fbank_bins = 128
    mel_lo_freq = 0
    mil_hi_freq = 8000
    use_htk = False
    n_FFT = 512

    audio_data = np.array(test_wav) / (2 ** 15)

    full_audio_data = np.tile(audio_data, 9)

    mfcc_params = preprocess.MFCCParams(samp_freq, num_fbank_bins, mel_lo_freq, mil_hi_freq, num_mfcc_feats,
                                          frame_len_samples, use_htk, n_FFT)
    mfcc_inst = preprocess.MFCC(mfcc_params)
    preprocessor = preprocess.Preprocessor(mfcc_inst, mode_input_size, stride)

    input_tensor = preprocessor.extract_features(full_audio_data)

    assert np.isclose(input_tensor[0][0], -3.4660944830426454)
    assert np.isclose(input_tensor[0][1], 0.3587718932127629)
    assert np.isclose(input_tensor[0][2], 0.3480551325669172)
    assert np.isclose(input_tensor[0][3], 0.2976191917228921)
    assert np.isclose(input_tensor[0][4], 0.3493037340849936)
    assert np.isclose(input_tensor[0][5], 0.2408643285767937)
    assert np.isclose(input_tensor[0][6], 0.2939659585037282)
    assert np.isclose(input_tensor[0][7], 0.2144552669573928)
    assert np.isclose(input_tensor[0][8], 0.302239565899944)
    assert np.isclose(input_tensor[0][9], 0.3187368787077345)
    assert np.isclose(input_tensor[0][10], 0.3019401051295793)
    assert np.isclose(input_tensor[0][11], 0.20449412797602678)

    assert np.isclose(input_tensor[0][38], -0.18751440767749533)


def test_sliding_window_for_wav_2_letter_sized_input():
    samp_freq = 16000
    frame_len_ms = 32
    frame_len_samples = samp_freq * frame_len_ms * 0.001
    num_mfcc_feats = 13
    mode_input_size = 296
    stride = 160
    num_fbank_bins = 128
    mel_lo_freq = 0
    mil_hi_freq = 8000
    use_htk = False
    n_FFT = 512

    audio_data = np.zeros(47712, dtype=int)

    mfcc_params = preprocess.MFCCParams(samp_freq, num_fbank_bins, mel_lo_freq, mil_hi_freq, num_mfcc_feats,
                                  frame_len_samples, use_htk, n_FFT)

    mfcc_inst = preprocess.MFCC(mfcc_params)
    preprocessor = preprocess.Preprocessor(mfcc_inst, mode_input_size, stride)

    input_tensor = preprocessor.extract_features(audio_data)

    assert len(input_tensor[0]) == 39
    assert len(input_tensor) == 296
