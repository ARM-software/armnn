//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include "Types.hpp"
#include "DetectedObject.hpp"
#include "IDetectionResultDecoder.hpp"

namespace od
{

class SSDResultDecoder : public IDetectionResultDecoder
{
public:
    /**
     * Constructs MobileNet ssd v1 inference results decoder.
     *
     * @param ObjectThreshold object score threshold
     */
    SSDResultDecoder(float ObjectThreshold);

    DetectedObjects Decode(const common::InferenceResults<float>& results,
                           const common::Size& outputFrameSize,
                           const common::Size& resizedFrameSize,
                           const std::vector<std::string>& labels) override;

private:
    float m_objectThreshold;
};
}// namespace od