//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include "Types.hpp"
#include "ArmnnNetworkExecutor.hpp"
#include "DetectedObject.hpp"
#include "IDetectionResultDecoder.hpp"
#include "NonMaxSuppression.hpp"

namespace od
{

class YoloResultDecoder : public IDetectionResultDecoder
{

public:
    /**
     * Constructs Yolo V3  inference reuslts decoder.
     *
     * @param NMSThreshold non max suppression threshold
     * @param ClsThreshold class probability threshold
     * @param ObjectThreshold detected object score threshold
     */
    YoloResultDecoder(float NMSThreshold, float ClsThreshold, float ObjectThreshold);

    DetectedObjects Decode(const common::InferenceResults<float>& results,
                           const common::Size& outputFrameSize,
                           const common::Size& resizedFrameSize,
                           const std::vector <std::string>& labels) override;
private:
    float m_NmsThreshold;
    float m_ClsThreshold;
    float m_objectThreshold;

    unsigned int m_boxElements = 4U;
    unsigned int m_confidenceElements = 1U;
    unsigned int m_numClasses = 80U;
    unsigned int m_numBoxes = 2535U;
};
}// namespace od