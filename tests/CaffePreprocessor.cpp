//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include "InferenceTestImage.hpp"
#include "CaffePreprocessor.hpp"

#include <boost/numeric/conversion/cast.hpp>
#include <boost/log/trivial.hpp>
#include <boost/assert.hpp>
#include <boost/format.hpp>

#include <iostream>
#include <fcntl.h>
#include <array>

const std::vector<ImageSet> g_DefaultImageSet =
{
    {"shark.jpg", 2}
};

CaffePreprocessor::CaffePreprocessor(const std::string& binaryFileDirectory, unsigned int width, unsigned int height,
                                   const std::vector<ImageSet>& imageSet)
:   m_BinaryDirectory(binaryFileDirectory)
,   m_Height(height)
,   m_Width(width)
,   m_ImageSet(imageSet.empty() ? g_DefaultImageSet : imageSet)
{
}

std::unique_ptr<CaffePreprocessor::TTestCaseData> CaffePreprocessor::GetTestCaseData(unsigned int testCaseId)
{
    testCaseId = testCaseId % boost::numeric_cast<unsigned int>(m_ImageSet.size());
    const ImageSet& imageSet = m_ImageSet[testCaseId];
    const std::string fullPath = m_BinaryDirectory + imageSet.first;

    InferenceTestImage image(fullPath.c_str());
    image.Resize(m_Width, m_Height, CHECK_LOCATION());

    // The model expects image data in BGR format.
    std::vector<float> inputImageData = GetImageDataInArmNnLayoutAsFloatsSubtractingMean(ImageChannelLayout::Bgr,
                                                                                         image, m_MeanBgr);

    // List of labels: https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a
    const unsigned int label = imageSet.second;
    return std::make_unique<TTestCaseData>(label, std::move(inputImageData));
}
