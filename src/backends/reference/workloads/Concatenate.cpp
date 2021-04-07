//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "Concatenate.hpp"
#include "RefWorkloadUtils.hpp"
#include "Decoders.hpp"
#include "Encoders.hpp"

namespace armnn
{

void Concatenate(const ConcatQueueDescriptor &data,
                 std::vector<ITensorHandle*> inputs,
                 std::vector<ITensorHandle*> outputs)
{
    const TensorInfo& outputInfo0 = GetTensorInfo(outputs[0]);

    std::unique_ptr<Encoder<float>> encoderPtr = MakeEncoder<float>(outputInfo0, outputs[0]->Map());
    Encoder<float>& encoder = *encoderPtr;

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
            ConcatQueueDescriptor::ViewOrigin const& view = data.m_ViewOrigins[viewIdx];

            //Split view extents are defined by the size of (the corresponding) input tensor.
            const TensorInfo& inputInfo = GetTensorInfo(inputs[viewIdx]);
            ARMNN_ASSERT(inputInfo.GetNumDimensions() == outputInfo0.GetNumDimensions());

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
                std::unique_ptr<Decoder<float>> decoderPtr =
                    MakeDecoder<float>(inputInfo,inputs[viewIdx]->Map());
                Decoder<float>& decoder = *decoderPtr;
                unsigned int inIndex = 0;
                unsigned int dimensionStride = 1;

                for (unsigned int i = inputInfo.GetNumDimensions(); i-- > 0;)
                {
                    inIndex += dimensionStride * (indices[i] - view.m_Origin[i]);
                    dimensionStride *= inputInfo.GetShape()[i];
                }
                decoder += inIndex;
                encoder.Set(decoder.Get());

                //What should we do if input views overlap on the output tensor?
                //We could error, take the average, or shm else...
                //For now just stop after finding first view (input) that matches.
                break;
            }
        }
        ++encoder;
    }
}

} //namespace armnn
