# Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
# SPDX-License-Identifier: MIT

"""Keyword Spotting with PyArmNN demo for processing live microphone data or pre-recorded files."""

import sys
import os
from argparse import ArgumentParser

import numpy as np
import sounddevice as sd

script_dir = os.path.dirname(__file__)
sys.path.insert(1, os.path.join(script_dir, '..', 'common'))

from network_executor import ArmnnNetworkExecutor
from utils import prepare_input_data, dequantize_output
from mfcc import AudioPreprocessor, MFCC, MFCCParams
from audio_utils import decode, display_text
from audio_capture import AudioCaptureParams, CaptureAudioStream, capture_audio

# Model Specific Labels
labels = {0: 'silence',
          1: 'unknown',
          2: 'yes',
          3: 'no',
          4: 'up',
          5: 'down',
          6: 'left',
          7: 'right',
          8: 'on',
          9: 'off',
          10: 'stop',
          11: 'go'}


def parse_args():
    parser = ArgumentParser(description="KWS with PyArmNN")
    parser.add_argument(
        "--audio_file_path",
        required=False,
        type=str,
        help="Path to the audio file to perform KWS",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=0,
        help="""Duration for recording audio in seconds. Values <= 0 result in infinite
           recording. Defaults to infinite.""",
    )
    parser.add_argument(
        "--model_file_path",
        required=True,
        type=str,
        help="Path to KWS model to use",
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


def recognise_speech(audio_data, network, preprocessor, threshold):
    # Prepare the input Tensors
    input_data = prepare_input_data(audio_data, network.get_data_type(), network.get_input_quantization_scale(0),
                                    network.get_input_quantization_offset(0), preprocessor)
    # Run inference
    output_result = network.run([input_data])

    dequantized_result = []
    for index, ofm in enumerate(output_result):
        dequantized_result.append(dequantize_output(ofm, network.is_output_quantized(index),
                                                    network.get_output_quantization_scale(index),
                                                    network.get_output_quantization_offset(index)))

    # Decode the text and display result if above threshold
    decoded_result = decode(dequantized_result, labels)

    if decoded_result[1] > threshold:
        display_text(decoded_result)


def main(args):
    # Read command line args and invoke mic streaming if no file path supplied
    audio_file = args.audio_file_path
    if args.audio_file_path:
        streaming_enabled = False
    else:
        streaming_enabled = True
    # Create the ArmNN inference runner
    network = ArmnnNetworkExecutor(args.model_file_path, args.preferred_backends)

    # Specify model specific audio data requirements
    # Overlap value specifies the number of samples to rewind between each data window
    audio_capture_params = AudioCaptureParams(dtype=np.float32, overlap=2000, min_samples=16000, sampling_freq=16000,
                                              mono=True)

    # Create the preprocessor
    mfcc_params = MFCCParams(sampling_freq=16000, num_fbank_bins=40, mel_lo_freq=20, mel_hi_freq=4000,
                             num_mfcc_feats=10, frame_len=640, use_htk_method=True, n_fft=1024)
    mfcc = MFCC(mfcc_params)
    preprocessor = AudioPreprocessor(mfcc, model_input_size=49, stride=320)

    # Set threshold for displaying classification and commence stream or file processing
    threshold = .90
    if streaming_enabled:
        # Initialise audio stream
        record_stream = CaptureAudioStream(audio_capture_params)
        record_stream.set_stream_defaults()
        record_stream.set_recording_duration(args.duration)
        record_stream.countdown()

        with sd.InputStream(callback=record_stream.callback):
            print("Recording audio. Please speak.")
            while record_stream.is_active:

                audio_data = record_stream.capture_data()
                recognise_speech(audio_data, network, preprocessor, threshold)
                record_stream.is_first_window = False
            print("\nFinished recording.")

    # If file path has been supplied read-in and run inference
    else:
        print("Processing Audio Frames...")
        buffer = capture_audio(audio_file, audio_capture_params)
        for audio_data in buffer:
            recognise_speech(audio_data, network, preprocessor, threshold)


if __name__ == "__main__":
    args = parse_args()
    main(args)
