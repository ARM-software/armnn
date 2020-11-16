# Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
# SPDX-License-Identifier: MIT

"""Automatic speech recognition with PyArmNN demo for processing audio clips to text."""

import sys
import os
from argparse import ArgumentParser

script_dir = os.path.dirname(__file__)
sys.path.insert(1, os.path.join(script_dir, '..', 'common'))

from network_executor import ArmnnNetworkExecutor
from utils import dict_labels
from preprocess import MFCCParams, Preprocessor, MFCC
from audio_capture import AudioCapture, ModelParams
from audio_utils import decode_text, prepare_input_tensors, display_text


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
        "--labels_file_path",
        required=True,
        type=str,
        help="Path to text file containing labels to map to model output",
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
    model = ModelParams(args.model_file_path)
    labels = dict_labels(args.labels_file_path)

    # Create the ArmNN inference runner
    network = ArmnnNetworkExecutor(model.path, args.preferred_backends)

    audio_capture = AudioCapture(model)
    buffer = audio_capture.from_audio_file(audio_file)

    # Create the preprocessor
    mfcc_params = MFCCParams(sampling_freq=16000, num_fbank_bins=128, mel_lo_freq=0, mel_hi_freq=8000,
                                        num_mfcc_feats=13, frame_len=512, use_htk_method=False, n_FFT=512)
    mfcc = MFCC(mfcc_params)
    preprocessor = Preprocessor(mfcc, model_input_size=1044, stride=160)

    text = ""
    current_r_context = ""
    is_first_window = True

    print("Processing Audio Frames...")
    for audio_data in buffer:
        # Prepare the input Tensors
        input_tensors = prepare_input_tensors(audio_data, network.input_binding_info, preprocessor)

        # Run inference
        output_result = network.run(input_tensors)

        # Slice and Decode the text, and store the right context
        current_r_context, text = decode_text(is_first_window, labels, output_result)

        is_first_window = False

        display_text(text)

    print(current_r_context, flush=True)


if __name__ == "__main__":
    args = parse_args()
    main(args)
