//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include "NonMaxSuppression.hpp"

#include <algorithm>

namespace od
{

static std::vector<unsigned int> GenerateRangeK(unsigned int k)
{
    std::vector<unsigned int> range(k);
    std::iota(range.begin(), range.end(), 0);
    return range;
}


/**
* @brief Returns the intersection over union for two bounding boxes
*
* @param[in]  First detect containing bounding box.
* @param[in]  Second detect containing bounding box.
* @return     Calculated intersection over union.
*
*/
static double IntersectionOverUnion(DetectedObject& detect1, DetectedObject& detect2)
{
    uint32_t area1 = (detect1.GetBoundingBox().GetHeight() * detect1.GetBoundingBox().GetWidth());
    uint32_t area2 = (detect2.GetBoundingBox().GetHeight() * detect2.GetBoundingBox().GetWidth());

    float yMinIntersection = std::max(detect1.GetBoundingBox().GetY(), detect2.GetBoundingBox().GetY());
    float xMinIntersection = std::max(detect1.GetBoundingBox().GetX(), detect2.GetBoundingBox().GetX());

    float yMaxIntersection = std::min(detect1.GetBoundingBox().GetY() + detect1.GetBoundingBox().GetHeight(),
                                      detect2.GetBoundingBox().GetY() + detect2.GetBoundingBox().GetHeight());
    float xMaxIntersection = std::min(detect1.GetBoundingBox().GetX() + detect1.GetBoundingBox().GetWidth(),
                                      detect2.GetBoundingBox().GetX() + detect2.GetBoundingBox().GetWidth());

    double areaIntersection = std::max(yMaxIntersection - yMinIntersection, 0.0f) *
                              std::max(xMaxIntersection - xMinIntersection, 0.0f);
    double areaUnion = area1 + area2 - areaIntersection;

    return areaIntersection / areaUnion;
}

std::vector<int> NonMaxSuppression(DetectedObjects& inputDetections, float iouThresh)
{
    // Sort indicies of detections by highest score to lowest.
    std::vector<unsigned int> sortedIndicies = GenerateRangeK(inputDetections.size());
    std::sort(sortedIndicies.begin(), sortedIndicies.end(),
        [&inputDetections](int idx1, int idx2)
        {
            return inputDetections[idx1].GetScore() > inputDetections[idx2].GetScore();
        });

    std::vector<bool> visited(inputDetections.size(), false);
    std::vector<int> outputIndiciesAfterNMS;

    for (int i=0; i < inputDetections.size(); ++i)
    {
        // Each new unvisited detect should be kept.
        if (!visited[sortedIndicies[i]])
        {
            outputIndiciesAfterNMS.emplace_back(sortedIndicies[i]);
            visited[sortedIndicies[i]] = true;
        }

        // Look for detections to suppress.
        for (int j=i+1; j<inputDetections.size(); ++j)
        {
            // Skip if already kept or suppressed.
            if (!visited[sortedIndicies[j]])
            {
                // Detects must have the same label to be suppressed.
                if (inputDetections[sortedIndicies[j]].GetLabel() == inputDetections[sortedIndicies[i]].GetLabel())
                {
                    auto iou = IntersectionOverUnion(inputDetections[sortedIndicies[i]],
                                                    inputDetections[sortedIndicies[j]]);
                    if (iou > iouThresh)
                    {
                        visited[sortedIndicies[j]] = true;
                    }
                }
            }
        }
    }
    return outputIndiciesAfterNMS;
}

} // namespace od
