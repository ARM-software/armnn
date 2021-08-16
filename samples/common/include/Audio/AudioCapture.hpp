//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <string>
#include <iostream>
#include <cmath>
#include <vector>
#include <exception>

#include "SlidingWindow.hpp"

namespace audio
{

/**
* @brief Class used to capture the audio data loaded from file, and to provide a method of
 * extracting correctly positioned and appropriately sized audio windows
*
*/
    class AudioCapture
    {
    public:

        SlidingWindow<const float> m_window;

        /**
        * @brief Default constructor
        */
        AudioCapture() = default;

        /**
        * @brief Function to load the audio data captured from the
         * input file to memory.
        */
        static std::vector<float> LoadAudioFile(std::string filePath);

        /**
        * @brief Function to initialize the sliding window. This will set its position in memory, its
         * window size and its stride.
        */
        void InitSlidingWindow(float* data, size_t dataSize, int minSamples, size_t stride);

        /**
        * Checks whether there is another block of audio in memory to read
        */
        bool HasNext();

        /**
        * Retrieves the next block of audio if its available
        */
        std::vector<float> Next();
    };
} // namespace audio