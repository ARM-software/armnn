//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "AudioCapture.hpp"
#include <alsa/asoundlib.h>
#include <sndfile.h>
#include <samplerate.h>

namespace audio
{
    std::vector<float> AudioCapture::LoadAudioFile(std::string filePath)
    {
        SF_INFO inputSoundFileInfo;
        SNDFILE* infile = nullptr;
        infile = sf_open(filePath.c_str(), SFM_READ, &inputSoundFileInfo);

        float audioIn[inputSoundFileInfo.channels * inputSoundFileInfo.frames];
        sf_read_float(infile, audioIn, inputSoundFileInfo.channels * inputSoundFileInfo.frames);

        float sampleRate = 16000.0f;
        float srcRatio = sampleRate / (float)inputSoundFileInfo.samplerate;
        int outputFrames = ceilf(inputSoundFileInfo.frames * srcRatio);

        // Convert to mono
        std::vector<float> monoData(inputSoundFileInfo.frames);
        for(int i = 0; i < inputSoundFileInfo.frames; i++)
        {
            for(int j = 0; j < inputSoundFileInfo.channels; j++)
                monoData[i] += audioIn[i * inputSoundFileInfo.channels + j];
            monoData[i] /= inputSoundFileInfo.channels;
        }

        // Resample
        SRC_DATA srcData;
        srcData.data_in = monoData.data();
        srcData.input_frames = inputSoundFileInfo.frames;

        std::vector<float> dataOut(outputFrames);
        srcData.data_out = dataOut.data();

        srcData.output_frames = outputFrames;
        srcData.src_ratio = srcRatio;

        src_simple(&srcData, SRC_SINC_BEST_QUALITY, 1);

        sf_close(infile);

        return dataOut;
    }

    void AudioCapture::InitSlidingWindow(float* data, size_t dataSize, int minSamples, size_t stride)
    {
        this->m_window = SlidingWindow<const float>(data, dataSize, minSamples, stride);
    }

    bool AudioCapture::HasNext()
    {
        return m_window.HasNext();
    }

    std::vector<float> AudioCapture::Next()
    {
        if (this->m_window.HasNext())
        {
            int remainingData = this->m_window.RemainingData();
            const float* windowData = this->m_window.Next();

            size_t windowSize = this->m_window.GetWindowSize();

            if(remainingData < windowSize)
            {
                std::vector<float> audioData(windowSize, 0.0f);
                for(int i = 0; i < remainingData; ++i)
                {
                    audioData[i] = *windowData;
                    if(i < remainingData - 1)
                    {
                        ++windowData;
                    }
                }
                return audioData;
            }
            else
            {
                std::vector<float> audioData(windowData, windowData + windowSize);
                return audioData;
            }
        }
        else
        {
            throw std::out_of_range("Error, end of audio data reached.");
        }
    }
} //namespace asr