//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "BoundingBox.hpp"

#include <string>
#include <vector>

namespace od
{
/**
 * An object detection network inference result decoded data representation.
 */
class DetectedObject
{

public:
    DetectedObject();

    /**
     * Creates detection with given parameters.
     *
     * @param id - class id
     * @param label - human readable text class label
     * @param boundingBox - rectangular detection coordinates
     * @param score - detection score/probability
     */
    DetectedObject(unsigned int id,
                   std::string  label,
                   const BoundingBox& boundingBox,
                   float score);

    ~DetectedObject() = default;

    /**
     * Get class id
     * @return id
     */
    unsigned int GetId() const;

    /**
     * Get human readable text class label
     * @return label
     */
    const std::string& GetLabel() const;

    /**
     * Get rectangular detection coordinates
     * @return detection coordinates
     */
    const BoundingBox& GetBoundingBox() const;

    /**
     * Get detection score
     * @return score
     */
    float GetScore() const;

    /**
     * Set class id
     * @param[in] id - class id
     */
    void SetId(unsigned int id);

    /**
     * Set class label
     * @param[in] label - human readable text class label
     */
    void SetLabel(const std::string& label);

    /**
     * Set detection coordinates
     * @param[in] boundingBox detection coordinates
     */
    void SetBoundingBox(const BoundingBox& boundingBox);

    /**
     * Set detection score
     * @param[in] score - detection score
     */
    void SetScore(float score);

private:
    unsigned int        m_Id;
    std::string         m_Label;
    BoundingBox         m_BoundingBox;
    float               m_Score;
};

using DetectedObjects = std::vector<DetectedObject>;

}// namespace od