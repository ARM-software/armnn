//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "Pad.hpp"
#include "backendsCommon/WorkloadData.hpp"
#include <boost/numeric/conversion/cast.hpp>
#include "TensorBufferArrayView.hpp"
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <cassert>

namespace armnn
{
template <typename T>
void Pad(const TensorInfo& inputInfo,
         const TensorInfo& outputInfo,
         std::vector<std::pair<unsigned int, unsigned int>> m_PadList,
         const T* inputData,
         T* outData)
{
    unsigned int numOutputElements = outputInfo.GetNumElements();

    TensorShape outputShape = outputInfo.GetShape();
    TensorShape inputShape = inputInfo.GetShape();

    unsigned int numInputDimensions = inputShape.GetNumDimensions();

    #ifndef NDEBUG

    unsigned int numOutputDimensions = outputShape.GetNumDimensions();
    assert(numInputDimensions == numOutputDimensions);

    #endif

    unsigned int inputBatches = 0;
    unsigned int inputChannels = 0;
    unsigned int inputHeight = 0;
    unsigned int inputWidth = 0;

    unsigned int outputChannels = 0;
    unsigned int outputHeight = 0;
    unsigned int outputWidth = 0;

    for (unsigned int i = 0; i < numOutputElements; ++i)
    {
       outData[i] = 0;
    }

    switch(numInputDimensions) {

        case 1:

            inputWidth = inputShape[0];

            for (unsigned int w = 0; w < inputWidth ; w++)
            {
                outData[w+std::get<0>(m_PadList[0])] = inputData[w];
            }

            break;

        case 2  :

            inputHeight = inputShape[0];
            inputWidth = inputShape[1];
            outputHeight = outputShape[0];
            outputWidth = outputShape[1];

            for (unsigned int h = 0; h < inputHeight; h++)
            {
                for (unsigned int w = 0; w < inputWidth ; w++)
                {
                    outData[(h+std::get<0>(m_PadList[0]))*outputWidth
                    + (w+std::get<0>(m_PadList[1]))] = inputData[h * inputWidth + w];
                }
            }

            break;

        case 3  :

            inputChannels = inputShape[0];
            inputHeight = inputShape[1];
            inputWidth = inputShape[2];
            outputChannels = outputShape[0];
            outputHeight = outputShape[1];
            outputWidth = outputShape[2];

            for (unsigned int c = 0; c < inputChannels; c++)
            {
                for (unsigned int h = 0; h < inputHeight; h++)
                {
                    for (unsigned int w = 0; w < inputWidth ; w++)
                    {
                        outData[(c+std::get<0>(m_PadList[0]))*outputHeight*outputWidth
                        + (h+std::get<0>(m_PadList[1]))*outputWidth
                        + (w+std::get<0>(m_PadList[2]))] = inputData[c * inputHeight * inputWidth
                                                                      + h * inputWidth
                                                                      + w];
                    }
                }
            }

            break;

        case 4  :

            inputBatches = inputShape[0];
            inputChannels = inputShape[1];
            inputHeight = inputShape[2];
            inputWidth = inputShape[3];
            outputChannels = outputShape[1];
            outputHeight = outputShape[2];
            outputWidth = outputShape[3];

            for (unsigned int b = 0; b < inputBatches; b++)
            {
                for (unsigned int c = 0; c < inputChannels; c++)
                {
                    for (unsigned int h = 0; h < inputHeight; h++)
                    {
                        for (unsigned int w = 0; w < inputWidth ; w++)
                        {
                            outData[(b+std::get<0>(m_PadList[0])) * outputChannels * outputHeight * outputWidth
                                   + (c+std::get<0>(m_PadList[1])) * outputHeight * outputWidth
                                   + (h+std::get<0>(m_PadList[2])) * outputWidth
                                   + (w+std::get<0>(m_PadList[3]))] = inputData[b * inputChannels * inputHeight
                                                                                * inputWidth
                                                                             + c * inputHeight * inputWidth
                                                                             + h * inputWidth
                                                                             + w];
                        }
                    }
                }
            }

            break;

        default :

            break;
    }
}

template void Pad<float>(const TensorInfo& inputInfo,
                         const TensorInfo& outputInfo,
                         std::vector<std::pair<unsigned int, unsigned int>> m_PadList,
                         const float* inputData,
                         float* outData);
template void Pad<uint8_t>(const TensorInfo& inputInfo,
                           const TensorInfo& outputInfo,
                           std::vector<std::pair<unsigned int, unsigned int>> m_PadList,
                           const uint8_t* inputData,
                           uint8_t* outData);

} //namespace armnn