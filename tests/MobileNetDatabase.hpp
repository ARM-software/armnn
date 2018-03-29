//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//
#pragma once

#include "ClassifierTestCaseData.hpp"

#include <array>
#include <string>
#include <vector>
#include <memory>

using ImageSet = std::pair<const std::string, unsigned int>;

class MobileNetDatabase
{
public:
    using TTestCaseData = ClassifierTestCaseData<float>;

    explicit MobileNetDatabase(const std::string& binaryFileDirectory,
        unsigned int width,
        unsigned int height,
        const std::vector<ImageSet>& imageSet);

    std::unique_ptr<TTestCaseData> GetTestCaseData(unsigned int testCaseId);

private:
    unsigned int GetNumImageElements() const { return 3 * m_Width * m_Height; }
    unsigned int GetNumImageBytes() const { return 4 * GetNumImageElements(); }

    std::string m_BinaryDirectory;
    unsigned int m_Height;
    unsigned int m_Width;
    const std::vector<ImageSet> m_ImageSet;
};