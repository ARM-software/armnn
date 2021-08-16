//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include <cmath>
#include <numeric>
#include <algorithm>
#include <memory>
#include "MathUtils.hpp"
#include "SlidingWindow.hpp"
#include "DsCNNPreprocessor.hpp"

std::vector<int8_t> kws::DsCNNPreprocessor::Invoke(const float* audioData, size_t dataSize,
                                                   int quantOffset, float quantScale) 
{
    auto window = SlidingWindow<const float>(
            audioData, dataSize,
            this->m_windowLen, this->m_windowStride);

    uint32_t mfccBufIdx = 0;
    std::vector<int8_t> outputBuffer;
    // While we can slide over the window
    while (window.HasNext()) 
    {
        const float* mfccWindow = window.Next();
        auto mfccAudioData = std::vector<float>(mfccWindow, mfccWindow + this->m_windowLen);

        auto mfcc = this->m_mfcc->MfccComputeQuant<int8_t>(mfccAudioData, quantScale, quantOffset);

        std::copy(mfcc.begin(), mfcc.end(), std::back_inserter(outputBuffer));

        ++mfccBufIdx;
    }

    return outputBuffer;
}

kws::DsCNNPreprocessor::DsCNNPreprocessor(const uint32_t windowLen, const uint32_t windowStride,
                                          std::unique_ptr<DsCnnMFCC> mfccInst) :
        m_windowLen{windowLen}, m_windowStride{windowStride}, m_mfcc{std::move(mfccInst)} {}
