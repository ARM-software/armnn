//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <string>
#include <utility>

namespace
{

struct BoundingBox
{
    BoundingBox()
        : BoundingBox(0.0f, 0.0f, 0.0f, 0.0f)
    {}

    BoundingBox(float xMin, float yMin, float xMax, float yMax)
        : m_XMin(xMin)
        , m_YMin(yMin)
        , m_XMax(xMax)
        , m_YMax(yMax)
    {}

    float m_XMin;
    float m_YMin;
    float m_XMax;
    float m_YMax;
};

struct DetectedObject
{
    DetectedObject(float detectedClass,
                   const BoundingBox& boundingBox,
                   float confidence)
        : m_Class(detectedClass)
        , m_BoundingBox(boundingBox)
        , m_Confidence(confidence)
    {}

    bool operator<(const DetectedObject& other) const
    {
        return m_Confidence < other.m_Confidence ||
            (m_Confidence == other.m_Confidence && m_Class < other.m_Class);
    }

    float        m_Class;
    BoundingBox  m_BoundingBox;
    float        m_Confidence;
};

using ObjectDetectionInput = std::pair<std::string, std::vector<DetectedObject>>;

} // anonymous namespace