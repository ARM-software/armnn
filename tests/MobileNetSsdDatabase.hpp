//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "InferenceTestImage.hpp"
#include "ObjectDetectionCommon.hpp"

#include <armnnUtils/QuantizeHelper.hpp>

#include <armnn/TypesUtils.hpp>
#include <armnn/utility/NumericCast.hpp>

#include <array>
#include <memory>
#include <string>
#include <vector>

namespace
{

struct MobileNetSsdTestCaseData
{
    MobileNetSsdTestCaseData(
        const std::vector<uint8_t>& inputData,
        const std::vector<DetectedObject>& expectedDetectedObject,
        const std::vector<std::vector<float>>& expectedOutput)
        : m_InputData(inputData)
        , m_ExpectedDetectedObject(expectedDetectedObject)
        , m_ExpectedOutput(expectedOutput)
    {}

    std::vector<uint8_t>            m_InputData;
    std::vector<DetectedObject>     m_ExpectedDetectedObject;
    std::vector<std::vector<float>> m_ExpectedOutput;
};

class MobileNetSsdDatabase
{
public:
    explicit MobileNetSsdDatabase(const std::string& imageDir, float scale, int offset);

    std::unique_ptr<MobileNetSsdTestCaseData> GetTestCaseData(unsigned int testCaseId);

private:
    std::string m_ImageDir;
    float m_Scale;
    int m_Offset;
};

constexpr unsigned int k_MobileNetSsdImageWidth  = 300u;
constexpr unsigned int k_MobileNetSsdImageHeight = k_MobileNetSsdImageWidth;

// Test cases
const std::array<ObjectDetectionInput, 1> g_PerTestCaseInput =
{
    ObjectDetectionInput
    {
        "Cat.jpg",
        {
          DetectedObject(16.0f, BoundingBox(0.216785252f, 0.079726994f, 0.927124202f, 0.939067304f), 0.79296875f)
        }
    }
};

MobileNetSsdDatabase::MobileNetSsdDatabase(const std::string& imageDir, float scale, int offset)
    : m_ImageDir(imageDir)
    , m_Scale(scale)
    , m_Offset(offset)
{}

std::unique_ptr<MobileNetSsdTestCaseData> MobileNetSsdDatabase::GetTestCaseData(unsigned int testCaseId)
{
    const unsigned int safeTestCaseId =
        testCaseId % armnn::numeric_cast<unsigned int>(g_PerTestCaseInput.size());
    const ObjectDetectionInput& testCaseInput = g_PerTestCaseInput[safeTestCaseId];

    // Load test case input
    const std::string imagePath = m_ImageDir + testCaseInput.first;
    std::vector<uint8_t> imageData;
    try
    {
        InferenceTestImage image(imagePath.c_str());

        // Resize image (if needed)
        const unsigned int width  = image.GetWidth();
        const unsigned int height = image.GetHeight();
        if (width != k_MobileNetSsdImageWidth || height != k_MobileNetSsdImageHeight)
        {
            image.Resize(k_MobileNetSsdImageWidth, k_MobileNetSsdImageHeight, CHECK_LOCATION());
        }

        // Get image data as a vector of floats
        std::vector<float> floatImageData = GetImageDataAsNormalizedFloats(ImageChannelLayout::Rgb, image);
        imageData = armnnUtils::QuantizedVector<uint8_t>(floatImageData, m_Scale, m_Offset);
    }
    catch (const InferenceTestImageException& e)
    {
        ARMNN_LOG(fatal) << "Failed to load image for test case " << testCaseId << ". Error: " << e.what();
        return nullptr;
    }

    std::vector<float> numDetections = { static_cast<float>(testCaseInput.second.size()) };

    std::vector<float> detectionBoxes;
    std::vector<float> detectionClasses;
    std::vector<float> detectionScores;

    for (DetectedObject expectedObject : testCaseInput.second)
    {
            detectionBoxes.push_back(expectedObject.m_BoundingBox.m_YMin);
            detectionBoxes.push_back(expectedObject.m_BoundingBox.m_XMin);
            detectionBoxes.push_back(expectedObject.m_BoundingBox.m_YMax);
            detectionBoxes.push_back(expectedObject.m_BoundingBox.m_XMax);

            detectionClasses.push_back(expectedObject.m_Class);

            detectionScores.push_back(expectedObject.m_Confidence);
    }

    // Prepare test case expected output
    std::vector<std::vector<float>> expectedOutputs;
    expectedOutputs.reserve(4);
    expectedOutputs.push_back(detectionBoxes);
    expectedOutputs.push_back(detectionClasses);
    expectedOutputs.push_back(detectionScores);
    expectedOutputs.push_back(numDetections);

    return std::make_unique<MobileNetSsdTestCaseData>(imageData, testCaseInput.second, expectedOutputs);
}

} // anonymous namespace
