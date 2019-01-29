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
        std::vector<float> inputData,
        std::vector<DetectedObject> expectedOutput)
        : m_InputData(std::move(inputData))
        , m_ExpectedOutput(std::move(expectedOutput))
    {}

    std::vector<float>          m_InputData;
    std::vector<DetectedObject> m_ExpectedOutput;
};

class MobileNetSsdDatabase
{
public:
    explicit MobileNetSsdDatabase(const std::string& imageDir);

    std::unique_ptr<MobileNetSsdTestCaseData> GetTestCaseData(unsigned int testCaseId);

private:
    std::string m_ImageDir;
};

constexpr unsigned int k_MobileNetSsdImageWidth  = 300u;
constexpr unsigned int k_MobileNetSsdImageHeight = k_MobileNetSsdImageWidth;

// Test cases
const std::array<ObjectDetectionInput, 1> g_PerTestCaseInput =
{
    ObjectDetectionInput
    {
        "Cat.jpg",
        DetectedObject(16, BoundingBox(0.21678525f, 0.0859828f, 0.9271242f, 0.9453231f), 0.79296875f)
    }
};

MobileNetSsdDatabase::MobileNetSsdDatabase(const std::string& imageDir)
    : m_ImageDir(imageDir)
{}

std::unique_ptr<MobileNetSsdTestCaseData> MobileNetSsdDatabase::GetTestCaseData(unsigned int testCaseId)
{
    const unsigned int safeTestCaseId =
        testCaseId % boost::numeric_cast<unsigned int>(g_PerTestCaseInput.size());
    const ObjectDetectionInput& testCaseInput = g_PerTestCaseInput[safeTestCaseId];

    // Load test case input
    const std::string imagePath = m_ImageDir + testCaseInput.first;
    std::vector<float> imageData;
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
        imageData = GetImageDataInArmNnLayoutAsNormalizedFloats(ImageChannelLayout::Rgb, image);
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
