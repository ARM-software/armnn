//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//

#pragma once

#include "RefWorkloadUtils.hpp"

#include "backends/WorkloadData.hpp"

#include <armnn/Tensor.hpp>

namespace armnn
{

template <typename DataType>
void Merger(const MergerQueueDescriptor& data)
{
    const TensorInfo& outputInfo0 = GetTensorInfo(data.m_Outputs[0]);

    for (unsigned int index = 0 ; index < outputInfo0.GetNumElements(); ++index)
    {
        unsigned int indices[MaxNumOfTensorDimensions] = { 0 };

        unsigned int indexRemainder = index;
        unsigned int dimensionStride = outputInfo0.GetNumElements();

        for (unsigned int i=0; i<outputInfo0.GetNumDimensions(); i++)
        {
            dimensionStride /= outputInfo0.GetShape()[i];
            indices[i] = indexRemainder / dimensionStride; // use integer division to round down
            indexRemainder -= indices[i] * dimensionStride;
        }

        for (unsigned int viewIdx = 0; viewIdx < data.m_ViewOrigins.size(); ++viewIdx)
        {
            MergerQueueDescriptor::ViewOrigin const& view = data.m_ViewOrigins[viewIdx];

            //split view extents are defined by the size of (the corresponding) input tensor
            const TensorInfo& inputInfo = GetTensorInfo(data.m_Inputs[viewIdx]);
            BOOST_ASSERT(inputInfo.GetNumDimensions() == outputInfo0.GetNumDimensions());

            // check all dimensions to see if this element is inside the given input view
            bool insideView = true;
            for (unsigned int i=0; i<inputInfo.GetNumDimensions(); i++)
            {
                if (indices[i] < view.m_Origin[i])
                {
                    insideView = false;
                }
                if (indices[i] >= view.m_Origin[i] + inputInfo.GetShape()[i])
                {
                    insideView = false;
                }
            }

            if (insideView)
            {
                unsigned int inIndex = 0;
                unsigned int dimensionStride = 1;

                for (unsigned int i = inputInfo.GetNumDimensions(); i-- > 0;)
                {
                    inIndex += dimensionStride * (indices[i] - view.m_Origin[i]);
                    dimensionStride *= inputInfo.GetShape()[i];
                }

                //we are within the view, copy input data to the output corresponding to this view
                (GetOutputTensorData<DataType>(0, data))[index] =
                    (GetInputTensorData<DataType>(viewIdx, data))[inIndex];

                //what should we do if input views overlap on the output tensor?
                //we could error, take the average, or shm else...
                //for now just stop after finding first view (input) that matches.
                break;
            }
        }
    }
}

} //namespace armnn
