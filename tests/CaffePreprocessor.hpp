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

/// Caffe requires BGR images, not normalized, mean adjusted and  resized using smooth resize of STB library

using ImageSet = std::pair<const std::string, unsigned int>;

class CaffePreprocessor
{
public:
    using DataType = float;
    using TTestCaseData = ClassifierTestCaseData<DataType>;

    explicit CaffePreprocessor(const std::string& binaryFileDirectory,
        unsigned int width = 227,
        unsigned int height = 227,
        const std::vector<ImageSet>& imageSet = std::vector<ImageSet>());
    std::unique_ptr<TTestCaseData> GetTestCaseData(unsigned int testCaseId);

private:
    unsigned int GetNumImageElements() const { return 3 * m_Width * m_Height; }
    unsigned int GetNumImageBytes() const { return 4 * GetNumImageElements(); }

    std::string m_BinaryDirectory;
    unsigned int m_Height;
    unsigned int m_Width;
    // Mean value of the database [B, G, R].
    const std::array<float, 3> m_MeanBgr = {{104.007965f, 116.669472f, 122.675102f}};
    const std::vector<ImageSet> m_ImageSet;
};
