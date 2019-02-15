//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include "ObjectDetectionCommon.hpp"

#include <memory>
#include <string>
#include <vector>

#include <armnn/TypesUtils.hpp>
#include <backendsCommon/test/QuantizeHelper.hpp>

#include <boost/log/trivial.hpp>
#include <boost/numeric/conversion/cast.hpp>

#include <array>
#include <string>

#include "InferenceTestImage.hpp"

namespace
{

struct MobileNetSsdTestCaseData
{
    MobileNetSsdTestCaseData(
        std::vector<uint8_t> inputData,
        std::vector<DetectedObject> expectedOutput)
        : m_InputData(std::move(inputData))
        , m_ExpectedOutput(std::move(expectedOutput))
    {}

    std::vector<uint8_t>        m_InputData;
    std::vector<DetectedObject> m_ExpectedOutput;
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
        DetectedObject(16, BoundingBox(0.208961248f, 0.0852333307f, 0.92757535f, 0.940263629f), 0.79296875f)
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
        testCaseId % boost::numeric_cast<unsigned int>(g_PerTestCaseInput.size());
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
        imageData = QuantizedVector<uint8_t>(m_Scale, m_Offset, floatImageData);
    }
    catch (const InferenceTestImageException& e)
    {
        BOOST_LOG_TRIVIAL(fatal) << "Failed to load image for test case " << testCaseId << ". Error: " << e.what();
        return nullptr;
    }

    // Prepare test case expected output
    std::vector<DetectedObject> expectedOutput;
    expectedOutput.reserve(1);
    expectedOutput.push_back(testCaseInput.second);

    return std::make_unique<MobileNetSsdTestCaseData>(std::move(imageData), std::move(expectedOutput));
}

} // anonymous namespace
