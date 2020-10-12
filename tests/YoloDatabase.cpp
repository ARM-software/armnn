//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include "YoloDatabase.hpp"

#include <armnn/Exceptions.hpp>
#include <armnn/Logging.hpp>

#include <armnn/utility/NumericCast.hpp>

#include <array>
#include <cstdint>
#include <tuple>
#include <utility>

#include "InferenceTestImage.hpp"

namespace
{
enum class YoloVocClass : unsigned int
{
    Aeroplane,
    Bicycle,
    Bird,
    Boat,
    Bottle,
    Bus,
    Car,
    Cat,
    Chair,
    Cow,
    DiningTable,
    Dog,
    Horse,
    Motorbike,
    Person,
    PottedPlant,
    Sheep,
    Sofa,
    Train,
    TvMonitor
};

template <typename E>
constexpr auto to_underlying(E e) noexcept
{
    return static_cast<std::underlying_type_t<E>>(e);
}

class ImageNotFoundException : public armnn::Exception
{
    using Exception::Exception;
};

using YoloInputOutput = std::pair<const char* const, YoloDetectedObject>;

const std::array<YoloInputOutput,1> g_PerTestCaseInputOutput =
{
    YoloInputOutput{
        "yolo_dog_448x448.png",
        { to_underlying(YoloVocClass::Dog), YoloBoundingBox{ 233.0f, 256.0f, 299.0f, 462.0f }, 0.5088733434677124f }
    },
};

} // namespace

YoloDatabase::YoloDatabase(const std::string& imageDir)
    : m_ImageDir(imageDir)
{
}

std::unique_ptr<YoloDatabase::TTestCaseData> YoloDatabase::GetTestCaseData(unsigned int testCaseId)
{
    testCaseId = testCaseId % armnn::numeric_cast<unsigned int>(g_PerTestCaseInputOutput.size());
    const auto& testCaseInputOutput = g_PerTestCaseInputOutput[testCaseId];
    const std::string imagePath = m_ImageDir + testCaseInputOutput.first;

    // Loads test case input image.
    std::vector<float> imageData;
    try
    {
        InferenceTestImage image(imagePath.c_str());
        if (YoloImageWidth != image.GetWidth() || YoloImageHeight != image.GetHeight())
        {
            image.Resize(YoloImageWidth, YoloImageHeight, CHECK_LOCATION());
        }
        imageData = GetImageDataInArmNnLayoutAsNormalizedFloats(ImageChannelLayout::Rgb, image);
    }
    catch (const InferenceTestImageException& e)
    {
        ARMNN_LOG(fatal) << "Failed to load test case " << testCaseId << " with error: " << e.what();
        return nullptr;
    }

    // Prepares test case output.
    std::vector<YoloDetectedObject> topObjectDetections;
    topObjectDetections.reserve(1);
    topObjectDetections.push_back(testCaseInputOutput.second);

    return std::make_unique<YoloTestCaseData>(std::move(imageData), std::move(topObjectDetections));
}
