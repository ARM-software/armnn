# Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
# SPDX-License-Identifier: MIT

"""Automatic speech recognition with PyArmNN demo for processing audio clips to text."""

import sys
import os
import numpy as np

script_dir = os.path.dirname(__file__)
sys.path.insert(1, os.path.join(script_dir, '..', 'common'))

from argparse import ArgumentParser
from network_executor import ArmnnNetworkExecutor
from utils import prepare_input_data
from audio_capture import AudioCaptureParams, capture_audio
from audio_utils import decode_text, display_text
from wav2letter_mfcc import Wav2LetterMFCC, W2LAudioPreprocessor
from mfcc import MFCCParams

# Model Specific Labels
labels = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h', 8: 'i', 9: 'j', 10: 'k', 11: 'l', 12: 'm',
          13: 'n',
          14: 'o', 15: 'p', 16: 'q', 17: 'r', 18: 's', 19: 't', 20: 'u', 21: 'v', 22: 'w', 23: 'x', 24: 'y',
          25: 'z',
          26: "'", 27: ' ', 28: '$'}


def parse_args():
    parser = ArgumentParser(description="ASR with PyArmNN")
    parser.add_argument(
        "--audio_file_path",
        required=True,
        type=str,
        help="Path to the audio file to perform ASR",
    )
    parser.add_argument(
        "--model_file_path",
        required=True,
        type=str,
        help="Path to ASR model to use",
    )
    parser.add_argument(
        "--preferred_backends",
        type=str,
        nargs="+",
        default=["CpuAcc", "CpuRef"],
        help="""List of backends in order of preference for optimizing
        subgraphs, falling back to the next backend in the list on unsupported
        layers. Defaults to [CpuAcc, CpuRef]""",
    )
    return parser.parse_args()


def main(args):
    # Read command line args
    audio_file = args.audio_file_path

    # Create the ArmNN inference runner
    network = ArmnnNetworkExecutor(args.model_file_path, args.preferred_backends)

    # Specify model specific audio data requirements
    audio_capture_params = AudioCaptureParams(dtype=np.float32, overlap=31712, min_samples=47712, sampling_freq=16000,
                                              mono=True)

    buffer = capture_audio(audio_file, audio_capture_params)

    # Extract features and create the preprocessor

    mfcc_params = MFCCParams(sampling_freq=16000, num_fbank_bins=128, mel_lo_freq=0, mel_hi_freq=8000,
                             num_mfcc_feats=13, frame_len=512, use_htk_method=False, n_fft=512)

    wmfcc = Wav2LetterMFCC(mfcc_params)
    preprocessor = W2LAudioPreprocessor(wmfcc, model_input_size=296, stride=160)
    current_r_context = ""
    is_first_window = True

    print("Processing Audio Frames...")
    for audio_data in buffer:
        # Prepare the input Tensors
        input_data = prepare_input_data(audio_data, network.get_data_type(), network.get_input_quantization_scale(0),
                                        network.get_input_quantization_offset(0), preprocessor)

        # Run inference
        output_result = network.run([input_data])

        # Slice and Decode the text, and store the right context
        current_r_context, text = decode_text(is_first_window, labels, output_result)

        is_first_window = False

        display_text(text)

    print(current_r_context, flush=True)


if __name__ == "__main__":
    args = parse_args()
    main(args)
