//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "RefWorkloadUtils.hpp"
#include <armnn/backends/WorkloadData.hpp>
#include <armnn/Tensor.hpp>
#include <armnn/utility/Assert.hpp>

namespace armnn
{

template <typename DataType>
void Splitter(const SplitterQueueDescriptor& data,
              std::vector<ITensorHandle*> inputs,
              std::vector<ITensorHandle*> outputs)
{
    const TensorInfo& inputInfo0 = GetTensorInfo(inputs[0]);

    for (unsigned int index = 0; index < inputInfo0.GetNumElements(); ++index)
    {
        unsigned int indices[MaxNumOfTensorDimensions] = { 0 };

        unsigned int indexRemainder = index;
        unsigned int dimensionStride = inputInfo0.GetNumElements();

        for (unsigned int i = 0; i<inputInfo0.GetNumDimensions(); i++)
        {
            dimensionStride /= inputInfo0.GetShape()[i];
            indices[i] = indexRemainder / dimensionStride; // Use integer division to round down.
            indexRemainder -= indices[i] * dimensionStride;
        }

        for (unsigned int viewIdx = 0; viewIdx < data.m_ViewOrigins.size(); ++viewIdx)
        {
            SplitterQueueDescriptor::ViewOrigin const& view = data.m_ViewOrigins[viewIdx];

            //Split view extents are defined by the size of (the corresponding) input tensor.
            const TensorInfo& outputInfo = GetTensorInfo(outputs[viewIdx]);
            ARMNN_ASSERT(outputInfo.GetNumDimensions() == inputInfo0.GetNumDimensions());

            // Check all dimensions to see if this element is inside the given input view.
            bool insideView = true;
            for (unsigned int i = 0; i<outputInfo.GetNumDimensions(); i++)
            {
                if (indices[i] < view.m_Origin[i])
                {
                    insideView = false;
                }
                if (indices[i] >= view.m_Origin[i] + outputInfo.GetShape()[i])
                {
                    insideView = false;
                }
            }

            if (insideView)
            {
                unsigned int outIndex = 0;
                unsigned int dimensionStride = 1;

                for (unsigned int i = outputInfo.GetNumDimensions(); i-- > 0;)
                {
                    outIndex += dimensionStride * (indices[i] - view.m_Origin[i]);
                    dimensionStride *= outputInfo.GetShape()[i];
                }

                //We are within the view, to copy input data to the output corresponding to this view.
                DataType* outputData = GetOutputTensorData<DataType>(viewIdx, data);
                ARMNN_ASSERT(outputData);

                const DataType* inputData = GetInputTensorData<DataType>(0, data);
                ARMNN_ASSERT(inputData);

                outputData[outIndex] = inputData[index];
            }
        }
    }
}

void Split(const SplitterQueueDescriptor& data,
           std::vector<ITensorHandle*> inputs,
           std::vector<ITensorHandle*> outputs);
} //namespace armnn
