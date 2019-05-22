//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include "ClassifierTestCaseData.hpp"

#include <array>
#include <string>
#include <vector>
#include <memory>

///Tf requires RGB images, normalized in range [0, 1] and resized using Bilinear algorithm


using ImageSet = std::pair<const std::string, unsigned int>;

template <typename TDataType>
class ImagePreprocessor
{
public:
    using DataType = TDataType;
    using TTestCaseData = ClassifierTestCaseData<DataType>;

    enum DataFormat
    {
        NHWC,
        NCHW
    };

    explicit ImagePreprocessor(const std::string& binaryFileDirectory,
        unsigned int width,
        unsigned int height,
        const std::vector<ImageSet>& imageSet,
        float scale=255.0f,
        const std::array<float, 3> mean={{0, 0, 0}},
        const std::array<float, 3> stddev={{1, 1, 1}},
        DataFormat dataFormat=DataFormat::NHWC,
        unsigned int batchSize=1)
    : m_BinaryDirectory(binaryFileDirectory)
    , m_Height(height)
    , m_Width(width)
    , m_BatchSize(batchSize)
    , m_Scale(scale)
    , m_ImageSet(imageSet)
    , m_Mean(mean)
    , m_Stddev(stddev)
    , m_DataFormat(dataFormat)
    {
    }

    std::unique_ptr<TTestCaseData> GetTestCaseData(unsigned int testCaseId);

private:
    unsigned int GetNumImageElements() const { return 3 * m_Width * m_Height; }
    unsigned int GetNumImageBytes() const { return sizeof(DataType) * GetNumImageElements(); }
    unsigned int GetLabelAndResizedImageAsFloat(unsigned int testCaseId,
                                                std::vector<float> & result);

    std::string m_BinaryDirectory;
    unsigned int m_Height;
    unsigned int m_Width;
    unsigned int m_BatchSize;
    // Quantization parameters
    float m_Scale;
    const std::vector<ImageSet> m_ImageSet;

    const std::array<float, 3> m_Mean;
    const std::array<float, 3> m_Stddev;

    DataFormat m_DataFormat;
};
