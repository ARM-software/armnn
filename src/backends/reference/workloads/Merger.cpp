//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "Merger.hpp"
#include "RefWorkloadUtils.hpp"

namespace armnn
{

template <>
void CopyValue<float>(const float& source, const TensorInfo& sourceInfo, float& dest, const TensorInfo& destInfo)
{
    dest = source;
}

template <>
void CopyValue<uint8_t>(const uint8_t& source, const TensorInfo& sourceInfo, uint8_t& dest, const TensorInfo& destInfo)
{
    if (sourceInfo.GetQuantizationScale() != destInfo.GetQuantizationScale() ||
        sourceInfo.GetQuantizationOffset() != destInfo.GetQuantizationOffset())
    {
        // Dequantize value acording to sourceInfo params
        float dequantizedValue = armnn::Dequantize<uint8_t>(source,
                                                            sourceInfo.GetQuantizationScale(),
                                                            sourceInfo.GetQuantizationOffset());

        // Quantize again according to destInfo paramns
        dest = armnn::Quantize<uint8_t>(dequantizedValue,
                destInfo.GetQuantizationScale(),
                destInfo.GetQuantizationOffset());
    }
    else
    {
        dest = source;
    }
}

template <typename DataType>
void Merger(const MergerQueueDescriptor& data)
{
    const TensorInfo& outputInfo0 = GetTensorInfo(data.m_Outputs[0]);

    for (unsigned int index = 0 ; index < outputInfo0.GetNumElements(); ++index)
    {
        unsigned int indices[MaxNumOfTensorDimensions] = { 0 };

        unsigned int indexRemainder = index;
        unsigned int dimensionStride = outputInfo0.GetNumElements();

        for (unsigned int i = 0; i < outputInfo0.GetNumDimensions(); i++)
        {
            dimensionStride /= outputInfo0.GetShape()[i];
            indices[i] = indexRemainder / dimensionStride; // Use integer division to round down.
            indexRemainder -= indices[i] * dimensionStride;
        }

        for (unsigned int viewIdx = 0; viewIdx < data.m_ViewOrigins.size(); ++viewIdx)
        {
            MergerQueueDescriptor::ViewOrigin const& view = data.m_ViewOrigins[viewIdx];

            //Split view extents are defined by the size of (the corresponding) input tensor.
            const TensorInfo& inputInfo = GetTensorInfo(data.m_Inputs[viewIdx]);
            BOOST_ASSERT(inputInfo.GetNumDimensions() == outputInfo0.GetNumDimensions());

            // Check all dimensions to see if this element is inside the given input view.
            bool insideView = true;
            for (unsigned int i = 0; i < inputInfo.GetNumDimensions(); i++)
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

                CopyValue<DataType>((GetInputTensorData<DataType>(viewIdx, data))[inIndex],
                                    GetTensorInfo(data.m_Inputs[viewIdx]),
                                    (GetOutputTensorData<DataType>(0, data))[index],
                                    outputInfo0);

                //What should we do if input views overlap on the output tensor?
                //We could error, take the average, or shm else...
                //For now just stop after finding first view (input) that matches.
                break;
            }
        }
    }
}

template void Merger<float>(const MergerQueueDescriptor& data);

template void Merger<uint8_t>(const MergerQueueDescriptor& data);

} //namespace armnn
