//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include "armnn/Tensor.hpp"
#include "armnn/Descriptors.hpp"

#include "Decoders.hpp"

#include <vector>

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
                          Decoder<float>& boxEncodings,
                          Decoder<float>& scores,
                          Decoder<float>& anchors,
                          float* detectionBoxes,
                          float* detectionClasses,
                          float* detectionScores,
                          float* numDetections);

void TopKSort(unsigned int k,
              unsigned int* indices,
              const float* values,
              unsigned int numElement);

float IntersectionOverUnion(const float* boxI, const float* boxJ);

std::vector<unsigned int> NonMaxSuppression(unsigned int numBoxes,
                                            const std::vector<float>& boxCorners,
                                            const std::vector<float>& scores,
                                            float nmsScoreThreshold,
                                            unsigned int maxDetection,
                                            float nmsIouThreshold);

} // namespace armnn
