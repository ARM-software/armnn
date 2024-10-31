//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "InferenceTestImage.hpp"
#include "ImagePreprocessor.hpp"

#include <armnn/TypesUtils.hpp>

#include <armnnUtils/Permute.hpp>
#include <armnn/utility/NumericCast.hpp>

#include <iostream>
#include <fcntl.h>
#include <array>

template <typename TDataType>
unsigned int ImagePreprocessor<TDataType>::GetLabelAndResizedImageAsFloat(unsigned int testCaseId,
                                                                          std::vector<float> & result)
{
    testCaseId = testCaseId % armnn::numeric_cast<unsigned int>(m_ImageSet.size());
    const ImageSet& imageSet = m_ImageSet[testCaseId];
    const std::string fullPath = m_BinaryDirectory + imageSet.first;

    InferenceTestImage image(fullPath.c_str());

    // this ResizeBilinear result is closer to the tensorflow one than STB.
    // there is still some difference though, but the inference results are
    // similar to tensorflow for MobileNet

    result = image.Resize(m_Width, m_Height, CHECK_LOCATION(),
                          InferenceTestImage::ResizingMethods::BilinearAndNormalized,
                          m_Mean, m_Stddev, m_Scale);

    // duplicate data across the batch
    for (unsigned int i = 1; i < m_BatchSize; i++)
    {
        result.insert(result.end(), result.begin(), result.begin() + armnn::numeric_cast<int>(GetNumImageElements()));
    }

    if (m_DataFormat == DataFormat::NCHW)
    {
        const armnn::PermutationVector NHWCToArmNN = { 0, 2, 3, 1 };
        armnn::TensorShape dstShape({m_BatchSize, 3, m_Height, m_Width});
        std::vector<float> tempImage(result.size());
        armnnUtils::Permute(dstShape, NHWCToArmNN, result.data(), tempImage.data(), sizeof(float));
        result.swap(tempImage);
    }

    return imageSet.second;
}

template <>
std::unique_ptr<ImagePreprocessor<float>::TTestCaseData>
ImagePreprocessor<float>::GetTestCaseData(unsigned int testCaseId)
{
    std::vector<float> resized;
    auto label = GetLabelAndResizedImageAsFloat(testCaseId, resized);
    return std::make_unique<TTestCaseData>(label, std::move(resized));
}

template <>
std::unique_ptr<ImagePreprocessor<uint8_t>::TTestCaseData>
ImagePreprocessor<uint8_t>::GetTestCaseData(unsigned int testCaseId)
{
    std::vector<float> resized;
    auto label = GetLabelAndResizedImageAsFloat(testCaseId, resized);

    size_t resizedSize = resized.size();
    std::vector<uint8_t> quantized(resized.size());

    for (size_t i=0; i<resizedSize; ++i)
    {
        quantized[i] = static_cast<uint8_t>(resized[i]);
    }

    return std::make_unique<TTestCaseData>(label, std::move(quantized));
}
