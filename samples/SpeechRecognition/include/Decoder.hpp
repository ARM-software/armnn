//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <string>
#include <map>
#include <vector>
#include <algorithm>
#include <cmath>

# pragma once

namespace asr
{
/**
* @brief Class used to Decode the output of the ASR inference
*
*/
    class Decoder
    {
    public:
        std::map<int, std::string> m_labels;
        /**
        * @brief Default constructor
        * @param[in] labels - map of labels to be used for decoding to text.
        */
        Decoder(std::map<int, std::string>& labels);

        /**
        * @brief Function to decode the output into a text string
        * @param[in] output - the output vector to decode.
        */
        template<typename T>
        std::string DecodeOutput(std::vector<T>& contextToProcess)
        {
            int rowLength = 29;

            std::vector<char> unfilteredText;

            for(int row = 0; row < contextToProcess.size()/rowLength; ++row)
            {
                std::vector<int16_t> rowVector;
                for(int j = 0; j < rowLength; ++j)
                {
                    rowVector.emplace_back(static_cast<int16_t>(contextToProcess[row * rowLength + j]));
                }

                int maxIndex = std::distance(rowVector.begin(), std::max_element(rowVector.begin(), rowVector.end()));
                unfilteredText.emplace_back(this->m_labels.at(maxIndex)[0]);
            }

            std::string filteredText = FilterCharacters(unfilteredText);
            return filteredText;
        }

        /**
        * @brief Function to filter out unwanted characters
        * @param[in] unfiltered - the unfiltered output to be processed.
        */
        std::string FilterCharacters(std::vector<char>& unfiltered);
    };
} // namespace asr
