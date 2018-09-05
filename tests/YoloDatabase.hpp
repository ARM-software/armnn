//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include "ClassifierTestCaseData.hpp"

#include <array>
#include <string>
#include <memory>

struct YoloBoundingBox
{
    float m_X;
    float m_Y;
    float m_W;
    float m_H;
};

struct YoloDetectedObject
{
    YoloDetectedObject(unsigned int yoloClass,
        const YoloBoundingBox& box,
        float confidence)
     : m_Class(yoloClass)
     , m_Box(box)
     , m_Confidence(confidence)
    {}

    unsigned int m_Class;
    YoloBoundingBox m_Box;
    float m_Confidence;
};

class YoloTestCaseData
{
public:
    YoloTestCaseData(std::vector<float> inputImage,
        std::vector<YoloDetectedObject> topObjectDetections)
     : m_InputImage(std::move(inputImage))
     , m_TopObjectDetections(std::move(topObjectDetections))
    {
    }

    std::vector<float> m_InputImage;
    std::vector<YoloDetectedObject> m_TopObjectDetections;
};

constexpr unsigned int YoloImageWidth = 448;
constexpr unsigned int YoloImageHeight = 448;

class YoloDatabase
{
public:
    using TTestCaseData = YoloTestCaseData;

    explicit YoloDatabase(const std::string& imageDir);
    std::unique_ptr<TTestCaseData> GetTestCaseData(unsigned int testCaseId);

private:
    std::string m_ImageDir;
};
