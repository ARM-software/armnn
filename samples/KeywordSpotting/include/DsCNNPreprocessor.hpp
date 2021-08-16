//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#ifndef KEYWORD_SPOTTING_EXAMPLE_DSCNNPREPROCESSOR_HPP
#define KEYWORD_SPOTTING_EXAMPLE_DSCNNPREPROCESSOR_HPP

#include <numeric>
#include "DsCnnMfcc.hpp"

namespace kws 
{
class DsCNNPreprocessor
{
public:
    DsCNNPreprocessor(uint32_t windowLen, uint32_t windowStride,
                      std::unique_ptr<DsCnnMFCC> mfccInst);

    /**
    * @brief       Calculates the features required from audio data. This
    *              includes MFCC, first and second order deltas,
    *              normalisation and finally, quantisation. The tensor is
    *              populated with feature from a given window placed along
    *              in a single row.
    * @param[in]   audioData     pointer to the first element of audio data
    * @param[in]   output        output to be populated
    * @return      true if successful, false in case of error.
    */
    std::vector<int8_t> Invoke(const float* audioData, 
                               size_t dataSize,
                               int quantOffset,
                               float quantScale) ;

    uint32_t m_windowLen;       // Window length for MFCC
    uint32_t m_windowStride;    // Window stride len for MFCC
    std::unique_ptr<MFCC> m_mfcc;
};
} // namespace kws
#endif //KEYWORD_SPOTTING_EXAMPLE_DSCNNPREPROCESSOR_HPP
