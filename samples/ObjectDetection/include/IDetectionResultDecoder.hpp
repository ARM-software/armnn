//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "DetectedObject.hpp"
#include "Types.hpp"

#include <vector>

namespace od
{

class IDetectionResultDecoder
{
public:
    /**
    * @brief    Returns decoded detected objects from a network model.
    * @desc     Outputs 4 vectors: bounding boxes, label, probabilities & number of detections.
    *           This function decodes network model output and converts it to expected format.
    *
    * @param[in]  results                 Vector of outputs from a model.
    * @param[in]  outputFrameSize         Struct containing height & width of output frame that is displayed.
    * @param[in]  resizedFrameSize        Struct containing height & width of resized input frame before padding
    * and inference.
    * @param[in]  labels                  Vector of network labels.
    * @param[in]  detectionScoreThreshold float value for the detection score threshold.
    *
    * @return     Vector of decoded detected objects.
    */
    virtual DetectedObjects Decode(const common::InferenceResults<float>& results,
                                   const common::Size& outputFrameSize,
                                   const common::Size& resizedFrameSize,
                                   const std::vector<std::string>& labels) = 0;

};
}// namespace od