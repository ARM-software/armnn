//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "YoloResultDecoder.hpp"

#include "NonMaxSuppression.hpp"

#include <cassert>
#include <stdexcept>

namespace od
{

DetectedObjects YoloResultDecoder::Decode(const common::InferenceResults<float>& networkResults,
                                         const common::Size& outputFrameSize,
                                         const common::Size& resizedFrameSize,
                                         const std::vector<std::string>& labels)
{

    // Yolo v3 network outputs 1 tensor
    if (networkResults.size() != 1)
    {
        throw std::runtime_error("Number of outputs from Yolo model doesn't equal 1");
    }
    auto element_step = m_boxElements + m_confidenceElements + m_numClasses;

    float longEdgeInput = std::max(resizedFrameSize.m_Width, resizedFrameSize.m_Height);
    float longEdgeOutput = std::max(outputFrameSize.m_Width, outputFrameSize.m_Height);
    const float resizeFactor = longEdgeOutput/longEdgeInput;

    DetectedObjects detectedObjects;
    DetectedObjects resultsAfterNMS;

    for (const common::InferenceResult<float>& result : networkResults)
    {
        for (unsigned int i = 0; i < m_numBoxes; ++i)
        {
            const float* cur_box = &result[i * element_step];
            // Objectness score
            if (cur_box[4] > m_objectThreshold)
            {
                for (unsigned int classIndex = 0; classIndex < m_numClasses; ++classIndex)
                {
                    const float class_prob =  cur_box[4] * cur_box[5 + classIndex];

                    // class confidence

                    if (class_prob > m_ClsThreshold)
                    {
                        DetectedObject detectedObject;

                        detectedObject.SetScore(class_prob);

                        float topLeftX = cur_box[0] * resizeFactor;
                        float topLeftY = cur_box[1] * resizeFactor;
                        float botRightX = cur_box[2] * resizeFactor;
                        float botRightY = cur_box[3] * resizeFactor;

                        assert(botRightX > topLeftX);
                        assert(botRightY > topLeftY);

                        detectedObject.SetBoundingBox({static_cast<int>(topLeftX),
                                                       static_cast<int>(topLeftY),
                                                       static_cast<unsigned int>(botRightX-topLeftX),
                                                       static_cast<unsigned int>(botRightY-topLeftY)});
                        if(labels.size() > classIndex)
                        {
                            detectedObject.SetLabel(labels.at(classIndex));
                        }
                        else
                        {
                            detectedObject.SetLabel(std::to_string(classIndex));
                        }
                        detectedObject.SetId(classIndex);
                        detectedObjects.emplace_back(detectedObject);
                    }
                }
            }
        }

        std::vector<int> keepIndiciesAfterNMS = od::NonMaxSuppression(detectedObjects, m_NmsThreshold);

        for (const int ind: keepIndiciesAfterNMS)
        {
            resultsAfterNMS.emplace_back(detectedObjects[ind]);
        }
    }

    return resultsAfterNMS;
}

YoloResultDecoder::YoloResultDecoder(float NMSThreshold, float ClsThreshold, float ObjectThreshold)
        : m_NmsThreshold(NMSThreshold), m_ClsThreshold(ClsThreshold), m_objectThreshold(ObjectThreshold) {}

}// namespace od



