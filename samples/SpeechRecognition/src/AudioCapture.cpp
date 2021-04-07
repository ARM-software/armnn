//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "AudioCapture.hpp"
#include <alsa/asoundlib.h>
#include <sndfile.h>
#include <samplerate.h>

namespace asr
{
    std::vector<float> AudioCapture::LoadAudioFile(std::string filePath)
    {
        SF_INFO inputSoundFileInfo;
        SNDFILE* infile = NULL;
        infile = sf_open(filePath.c_str(), SFM_READ, &inputSoundFileInfo);

        float audioIn[inputSoundFileInfo.channels * inputSoundFileInfo.frames];
        sf_read_float(infile, audioIn, inputSoundFileInfo.channels * inputSoundFileInfo.frames);

        float sampleRate = 16000.0f;
        float srcRatio = sampleRate / (float)inputSoundFileInfo.samplerate;
        int outputFrames = ceil(inputSoundFileInfo.frames * srcRatio);
        float dataOut[outputFrames];

        // Convert to mono
        float monoData[inputSoundFileInfo.frames];
        for(int i = 0; i < inputSoundFileInfo.frames; i++)
        {
            float val = 0.0f;
            for(int j = 0; j < inputSoundFileInfo.channels; j++)
                monoData[i] += audioIn[i * inputSoundFileInfo.channels + j];
            monoData[i] /= inputSoundFileInfo.channels;
        }

        // Resample
        SRC_DATA srcData;
        srcData.data_in = monoData;
        srcData.input_frames = inputSoundFileInfo.frames;
        srcData.data_out = dataOut;
        srcData.output_frames = outputFrames;
        srcData.src_ratio = srcRatio;

        src_simple(&srcData, SRC_SINC_BEST_QUALITY, 1);

        // Convert to Vector
        std::vector<float> processedInput;

        for(int i = 0; i < srcData.output_frames_gen; ++i)
        {
            processedInput.push_back(srcData.data_out[i]);
        }

        sf_close(infile);

        return processedInput;
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
                std::vector<float> mfccAudioData(windowSize, 0.0f);
                for(int i = 0; i < remainingData; ++i)
                {
                    mfccAudioData[i] = *windowData;
                    if(i < remainingData - 1)
                    {
                        ++windowData;
                    }
                }
                return mfccAudioData;
            }
            else
            {
                std::vector<float> mfccAudioData(windowData,  windowData + windowSize);
                return mfccAudioData;
            }
        }
        else
        {
            throw std::out_of_range("Error, end of audio data reached.");
        }
    }
} //namespace asr

