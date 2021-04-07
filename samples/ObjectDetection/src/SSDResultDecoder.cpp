//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "SSDResultDecoder.hpp"

#include <cassert>
#include <algorithm>
#include <cmath>
#include <stdexcept>
namespace od
{

DetectedObjects SSDResultDecoder::Decode(const common::InferenceResults<float>& networkResults,
    const common::Size& outputFrameSize,
    const common::Size& resizedFrameSize,
    const std::vector<std::string>& labels)
{
    // SSD network outputs 4 tensors: bounding boxes, labels, probabilities, number of detections.
    if (networkResults.size() != 4)
    {
        throw std::runtime_error("Number of outputs from SSD model doesn't equal 4");
    }

    DetectedObjects detectedObjects;
    const int numDetections = static_cast<int>(std::lround(networkResults[3][0]));

    double longEdgeInput = std::max(resizedFrameSize.m_Width, resizedFrameSize.m_Height);
    double longEdgeOutput = std::max(outputFrameSize.m_Width, outputFrameSize.m_Height);
    const double resizeFactor = longEdgeOutput/longEdgeInput;

    for (int i=0; i<numDetections; ++i)
    {
        if (networkResults[2][i] > m_objectThreshold)
        {
            DetectedObject detectedObject;
            detectedObject.SetScore(networkResults[2][i]);
            auto classId = std::lround(networkResults[1][i]);

            if (classId < labels.size())
            {
                detectedObject.SetLabel(labels[classId]);
            }
            else
            {
                detectedObject.SetLabel(std::to_string(classId));
            }
            detectedObject.SetId(classId);

            // Convert SSD bbox outputs (ratios of image size) to pixel values.
            double topLeftY = networkResults[0][i*4 + 0] * resizedFrameSize.m_Height;
            double topLeftX = networkResults[0][i*4 + 1] * resizedFrameSize.m_Width;
            double botRightY = networkResults[0][i*4 + 2] * resizedFrameSize.m_Height;
            double botRightX = networkResults[0][i*4 + 3] * resizedFrameSize.m_Width;

            // Scale the coordinates to output frame size.
            topLeftY *= resizeFactor;
            topLeftX *= resizeFactor;
            botRightY *= resizeFactor;
            botRightX *= resizeFactor;

            assert(botRightX > topLeftX);
            assert(botRightY > topLeftY);

            // Internal BoundingBox stores box top left x,y and width, height.
            detectedObject.SetBoundingBox({static_cast<int>(std::round(topLeftX)),
                                           static_cast<int>(std::round(topLeftY)),
                                           static_cast<unsigned int>(botRightX - topLeftX),
                                           static_cast<unsigned int>(botRightY - topLeftY)});

            detectedObjects.emplace_back(detectedObject);
        }
    }
    return detectedObjects;
}

SSDResultDecoder::SSDResultDecoder(float ObjectThreshold) : m_objectThreshold(ObjectThreshold) {}

}// namespace od