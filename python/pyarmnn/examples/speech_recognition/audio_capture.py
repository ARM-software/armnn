# Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
# SPDX-License-Identifier: MIT

"""Contains AudioCapture class for capturing chunks of audio data from file."""

from typing import Generator

import numpy as np
import soundfile as sf


class ModelParams:
    def __init__(self, model_file_path: str):
        """Defines sampling parameters for model used.

        Args:
            model_file_path: Path to ASR model to use.
        """
        self.path = model_file_path
        self.mono = True
        self.dtype = np.float32
        self.samplerate = 16000
        self.min_samples = 167392


class AudioCapture:
    def __init__(self, model_params):
        """Sampling parameters for model used."""
        self.model_params = model_params

    def from_audio_file(self, audio_file_path, overlap=31712) -> Generator[np.ndarray, None, None]:
        """Creates a generator that yields audio data from a file. Data is padded with
        zeros if necessary to make up minimum number of samples.

        Args:
            audio_file_path: Path to audio file provided by user.
            overlap: The overlap with previous buffer. We need the offset to be the same as the inner context
                    of the mfcc output, which is sized as 100 x 39. Each mfcc compute produces 1 x 39 vector,
                    and consumes 160 audio samples. The default overlap is then calculated to be 47712 - (160 x 100)
                    where 47712 is the min_samples needed for 1 inference of wav2letter.

        Yields:
            Blocks of audio data of minimum sample size.
        """
        with sf.SoundFile(audio_file_path) as audio_file:
            for block in audio_file.blocks(
                    blocksize=self.model_params.min_samples,
                    dtype=self.model_params.dtype,
                    always_2d=True,
                    fill_value=0,
                    overlap=overlap
            ):
                # Convert to mono if specified
                if self.model_params.mono and block.shape[0] > 1:
                    block = np.mean(block, axis=1)
                yield block
