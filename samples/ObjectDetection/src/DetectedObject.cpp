//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "DetectedObject.hpp"

namespace od
{

DetectedObject::DetectedObject() :
        DetectedObject(0u, "", BoundingBox(), 0u)
{}

DetectedObject::DetectedObject(
        unsigned int id,
        std::string label,
        const BoundingBox &boundingBox,
        float score) :
        m_Id(id),
        m_Label(std::move(label)),
        m_BoundingBox(boundingBox),
        m_Score(score)
{}

unsigned int DetectedObject::GetId() const
{
    return m_Id;
}

const std::string &DetectedObject::GetLabel() const
{
    return m_Label;
}

const BoundingBox &DetectedObject::GetBoundingBox() const
{
    return m_BoundingBox;
}

float DetectedObject::GetScore() const
{
    return m_Score;
}

void DetectedObject::SetId(unsigned int id)
{
    m_Id = id;
}

void DetectedObject::SetLabel(const std::string &label)
{
    m_Label = label;
}

void DetectedObject::SetBoundingBox(const BoundingBox &boundingBox)
{
    m_BoundingBox = boundingBox;
}

void DetectedObject::SetScore(float score)
{
    m_Score = score;
}
}// namespace od