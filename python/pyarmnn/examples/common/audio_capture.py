# Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
# SPDX-License-Identifier: MIT
"""Contains CaptureAudioStream class for capturing chunks of audio data from incoming
  stream and generic capture_audio function for capturing from files."""
import collections
import time
from queue import Queue
from typing import Generator

import numpy as np
import sounddevice as sd
import soundfile as sf

AudioCaptureParams = collections.namedtuple('AudioCaptureParams',
                                            ['dtype', 'overlap', 'min_samples', 'sampling_freq', 'mono'])


def capture_audio(audio_file_path, params_tuple) -> Generator[np.ndarray, None, None]:
    """Creates a generator that yields audio data from a file. Data is padded with
    zeros if necessary to make up minimum number of samples.
    Args:
        audio_file_path: Path to audio file provided by user.
        params_tuple: Sampling parameters for model used
    Yields:
        Blocks of audio data of minimum sample size.
    """
    with sf.SoundFile(audio_file_path) as audio_file:
        for block in audio_file.blocks(
                blocksize=params_tuple.min_samples,
                dtype=params_tuple.dtype,
                always_2d=True,
                fill_value=0,
                overlap=params_tuple.overlap
        ):
            if params_tuple.mono and block.shape[0] > 1:
                block = np.mean(block, dtype=block.dtype, axis=1)
            yield block


class CaptureAudioStream:

    def __init__(self, audio_capture_params):
        self.audio_capture_params = audio_capture_params
        self.collection = np.zeros(self.audio_capture_params.min_samples + self.audio_capture_params.overlap).astype(
            dtype=self.audio_capture_params.dtype)
        self.is_active = True
        self.is_first_window = True
        self.duration = False
        self.block_count = 0
        self.current_block = 0
        self.queue = Queue(2)

    def set_stream_defaults(self):
        """Discovers input devices on the system and sets default stream parameters."""
        print(sd.query_devices())
        device = input("Select input device by index or name: ")

        try:
            sd.default.device = int(device)
        except ValueError:
            sd.default.device = str(device)

        sd.default.samplerate = self.audio_capture_params.sampling_freq
        sd.default.blocksize = self.audio_capture_params.min_samples
        sd.default.dtype = self.audio_capture_params.dtype
        sd.default.channels = 1 if self.audio_capture_params.mono else 2

    def set_recording_duration(self, duration):
        """Sets a time duration (in integer seconds) for recording audio. Total time duration is
        adjusted to a minimum based on the parameters of the model used. Durations less than 1
        result in endless recording.

        Args:
            duration (int): User-provided command line argument for time duration of recording.
        """
        if duration > 0:
            min_duration = int(
                np.ceil(self.audio_capture_params.min_samples / self.audio_capture_params.sampling_freq)
            )
            if duration < min_duration:
                print(f"Minimum duration must be {min_duration} seconds of audio")
                print(f"Setting minimum recording duration...")
                duration = min_duration

            print(f"Recording duration is {duration} seconds")
            self.duration = self.audio_capture_params.sampling_freq * duration
            self.block_count, remainder_samples = divmod(
                self.duration, self.audio_capture_params.min_samples
            )

            if remainder_samples > 0.5 * self.audio_capture_params.sampling_freq:
                self.block_count += 1
        else:
            self.duration = False  # Record forever

    def countdown(self, delay=3):
        """3 second countdown prior to recording audio."""
        print("Beginning recording in...")
        for i in range(delay, 0, -1):
            print(f"{i}...")
            time.sleep(1)

    def update(self):
        """If a duration has been set, increments a counter to update the number of blocks of audio
        data left to be collected. The stream is deactivated upon reaching the maximum block count
        determined by the duration.
        """
        if self.duration:
            self.current_block += 1
            if self.current_block == self.block_count:
                self.is_active = False

    def capture_data(self):
        """Gets the next window of audio data by retrieving the newest data from a queue and
        shifting the position of the data in the collection. Overlap values of less than `min_samples` are supported.
        """
        new_data = self.queue.get()

        if self.is_first_window or self.audio_capture_params.overlap == 0:
            self.collection[:self.audio_capture_params.min_samples] = new_data[:]

        elif self.audio_capture_params.overlap < self.audio_capture_params.min_samples:
            #
            self.collection[0:self.audio_capture_params.overlap] = \
                self.collection[(self.audio_capture_params.min_samples - self.audio_capture_params.overlap):
                                self.audio_capture_params.min_samples]

            self.collection[self.audio_capture_params.overlap:(
                    self.audio_capture_params.overlap + self.audio_capture_params.min_samples)] = new_data[:]
        else:
            raise ValueError(
                "Capture Error: Overlap must be less than {}".format(self.audio_capture_params.min_samples))
        audio_data = self.collection[0:self.audio_capture_params.min_samples]
        return np.asarray(audio_data).astype(self.audio_capture_params.dtype)

    def callback(self, data, frames, time, status):
        """Places audio data from active stream into a queue for processing.
        Update counter if recording duration is finite.
         """

        if self.duration:
            self.update()

        if self.audio_capture_params.mono:
            audio_data = data.copy().flatten()
        else:
            audio_data = data.copy()

        self.queue.put(audio_data)
