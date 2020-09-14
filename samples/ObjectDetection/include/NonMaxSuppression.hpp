//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include "DetectedObject.hpp"

#include <numeric>
#include <vector>

namespace od
{

/**
* @brief Returns a vector of indices corresponding to input detections kept after NMS.
*
* Perform non max suppression on input detections. Any detections with iou greater than
* given threshold are suppressed. Different detection labels are considered independently.
*
* @param[in]  Vector of decoded detections.
* @param[in]  Detects with IOU larger than this threshold are suppressed.
* @return     Vector of indices corresponding to input detections kept after NMS.
*
*/
std::vector<int> NonMaxSuppression(DetectedObjects& inputDetections, float iouThresh);

}// namespace od