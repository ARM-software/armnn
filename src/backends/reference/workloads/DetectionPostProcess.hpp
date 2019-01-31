//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include "armnn/Tensor.hpp"
#include "armnn/Descriptors.hpp"

namespace armnn
{

void DetectionPostProcess(const TensorInfo& boxEncodingsInfo,
                          const TensorInfo& scoresInfo,
                          const TensorInfo& anchorsInfo,
                          const TensorInfo& detectionBoxesInfo,
                          const TensorInfo& detectionClassesInfo,
                          const TensorInfo& detectionScoresInfo,
                          const TensorInfo& numDetectionsInfo,
                          const DetectionPostProcessDescriptor& desc,
                          const float* boxEncodings,
                          const float* scores,
                          const float* anchors,
                          float* detectionBoxes,
                          float* detectionClasses,
                          float* detectionScores,
                          float* numDetections);

} // namespace armnn
