//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//
#include "InferenceTestImage.hpp"
#include "ImageNetDatabase.hpp"

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

ImageNetDatabase::ImageNetDatabase(const std::string& binaryFileDirectory, unsigned int width, unsigned int height,
                                   const std::vector<ImageSet>& imageSet)
:   m_BinaryDirectory(binaryFileDirectory)
,   m_Height(height)
,   m_Width(width)
,   m_ImageSet(imageSet.empty() ? g_DefaultImageSet : imageSet)
{
}

std::unique_ptr<ImageNetDatabase::TTestCaseData> ImageNetDatabase::GetTestCaseData(unsigned int testCaseId)
{
    testCaseId = testCaseId % boost::numeric_cast<unsigned int>(m_ImageSet.size());
    const ImageSet& imageSet = m_ImageSet[testCaseId];
    const std::string fullPath = m_BinaryDirectory + imageSet.first;

    InferenceTestImage image(fullPath.c_str());
    image.Resize(m_Width, m_Height);

    // The model expects image data in BGR format
    std::vector<float> inputImageData = GetImageDataInArmNnLayoutAsFloatsSubtractingMean(ImageChannelLayout::Bgr,
                                                                                         image, m_MeanBgr);

    // list of labels: https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a
    const unsigned int label = imageSet.second;
    return std::make_unique<TTestCaseData>(label, std::move(inputImageData));
}
