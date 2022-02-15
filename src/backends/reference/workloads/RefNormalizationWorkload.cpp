//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "RefNormalizationWorkload.hpp"

#include <armnn/Logging.hpp>
#include <armnn/Tensor.hpp>
#include <armnnUtils/DataLayoutIndexed.hpp>
#include <armnn/utility/NumericCast.hpp>

#include <Profiling.hpp>

#include "RefWorkloadUtils.hpp"
#include "Decoders.hpp"
#include "Encoders.hpp"

using namespace armnn;
using namespace armnnUtils;

namespace
{

// Helper function to compute "Within" normalization using Krichevsky 2012: Local Brightness Normalization.
void NormalizeWithinUingLbr(Decoder<float>&    inputData,
                            Encoder<float>&    outputData,
                            const TensorShape& tensorShape,
                            uint32_t           norm_size,
                            float              alpha,
                            float              beta,
                            float              kappa)
{
    const unsigned int batchSize = tensorShape[0];
    const unsigned int depth = tensorShape[1];
    const unsigned int rows = tensorShape[2];
    const unsigned int cols = tensorShape[3];

    int radius = armnn::numeric_cast<int>(norm_size / 2u); /* Strong Assumption on rounding Mode */

    for (unsigned int n = 0; n < batchSize; n++)
    {
        for (unsigned int c = 0; c < depth; c++)
        {
            for (unsigned int h = 0; h < rows; h++)
            {
                for (unsigned int w = 0; w < cols; w++)
                {
                    float accumulated_scale = 0.0;
                    for (int y = -radius; y <= radius; y++)
                    {
                        for (int x = -radius; x <= radius; x++)
                        {
                            int i = armnn::numeric_cast<int>(w) + x;
                            int j = armnn::numeric_cast<int>(h) + y;

                            if ((i < 0) || (i >= armnn::numeric_cast<int>(cols)))
                            {
                                continue;
                            }

                            if ((j < 0) || (j >= armnn::numeric_cast<int>(rows)))
                            {
                                continue;
                            }

                            unsigned int inputIndex = n * cols * rows * depth +
                                                      c * cols * rows +
                                                      armnn::numeric_cast<unsigned int>(j) * cols +
                                                      armnn::numeric_cast<unsigned int>(i);
                            inputData[inputIndex];
                            float inval = inputData.Get();

                            accumulated_scale += inval*inval;
                        }
                    }

                    unsigned int index = n * cols * rows * depth +
                                         c * cols * rows +
                                         h * cols +
                                         w;
                    inputData[index];
                    outputData[index];
                    outputData.Set(inputData.Get() / (powf((kappa + (accumulated_scale * alpha)), beta)));
                }
            }
        }
    }
}

// Helper function to compute "Across" normalization using Krichevsky 2012: Local Brightness Normalization.
void NormalizeAcrossUingLbr(Decoder<float>&    inputData,
                            Encoder<float>&    outputData,
                            const TensorShape& tensorShape,
                            uint32_t           norm_size,
                            float              alpha,
                            float              beta,
                            float              kappa,
                            DataLayout         dataLayout)
{
    DataLayoutIndexed dataLayoutIndexed(dataLayout);

    const unsigned int batchSize = tensorShape[0];
    const unsigned int depth     = tensorShape[dataLayoutIndexed.GetChannelsIndex()];
    const unsigned int rows      = tensorShape[dataLayoutIndexed.GetHeightIndex()];
    const unsigned int cols      = tensorShape[dataLayoutIndexed.GetWidthIndex()];

    int radius = armnn::numeric_cast<int>(norm_size / 2u); /* Strong Assumption on rounding Mode */

    for (unsigned int n = 0; n < batchSize; n++)
    {
        for (unsigned int c = 0; c < depth; c++)
        {
            for (unsigned int h = 0; h < rows; h++)
            {
                for (unsigned int w = 0; w < cols; w++)
                {
                    float accumulated_scale = 0.0;
                    for (int z = -radius; z <= radius; z++)
                    {
                        int k = armnn::numeric_cast<int>(c) + z;

                        if ((k < 0) || (k >= armnn::numeric_cast<int>(depth)))
                        {
                            continue;
                        }

                        unsigned inputIndex = dataLayoutIndexed.GetIndex(tensorShape,
                                                                         n,
                                                                         armnn::numeric_cast<unsigned int>(k),
                                                                         h,
                                                                         w);

                        inputData[inputIndex];
                        float inval = inputData.Get();

                        accumulated_scale += inval * inval;
                    }

                    float scale = kappa + (accumulated_scale * alpha);
                    scale = powf(scale, -beta);

                    unsigned index = dataLayoutIndexed.GetIndex(tensorShape, n, c, h, w);

                    inputData[index];
                    outputData[index];
                    outputData.Set(scale * inputData.Get());
                }
            }
        }
    }
}

} // Anonymous namespace

namespace armnn
{

RefNormalizationWorkload::RefNormalizationWorkload(const NormalizationQueueDescriptor& descriptor,
                                                   const WorkloadInfo& info)
    : RefBaseWorkload(descriptor, info)
{}

void RefNormalizationWorkload::Execute() const
{
    Execute(m_Data.m_Inputs, m_Data.m_Outputs);
}

void RefNormalizationWorkload::ExecuteAsync(WorkingMemDescriptor &workingMemDescriptor)
{
    Execute(workingMemDescriptor.m_Inputs, workingMemDescriptor.m_Outputs);
}

void RefNormalizationWorkload::Execute(std::vector<ITensorHandle*> inputs, std::vector<ITensorHandle*> outputs) const
{
    ARMNN_SCOPED_PROFILING_EVENT(Compute::CpuRef, "RefNormalizationWorkload_Execute");

    const TensorInfo& inputInfo = GetTensorInfo(inputs[0]);

    auto inputDecoder  = MakeDecoder<float>(inputInfo, inputs[0]->Map());
    auto outputEncoder = MakeEncoder<float>(inputInfo, outputs[0]->Map());

    if (NormalizationAlgorithmMethod::LocalBrightness == m_Data.m_Parameters.m_NormMethodType)
    {
        if (NormalizationAlgorithmChannel::Within == m_Data.m_Parameters.m_NormChannelType)
        {
            NormalizeWithinUingLbr(*inputDecoder,
                                   *outputEncoder,
                                   inputInfo.GetShape(),
                                   m_Data.m_Parameters.m_NormSize,
                                   m_Data.m_Parameters.m_Alpha,
                                   m_Data.m_Parameters.m_Beta,
                                   m_Data.m_Parameters.m_K);
        }
        else if (NormalizationAlgorithmChannel::Across == m_Data.m_Parameters.m_NormChannelType)
        {
            NormalizeAcrossUingLbr(*inputDecoder,
                                   *outputEncoder,
                                   inputInfo.GetShape(),
                                   m_Data.m_Parameters.m_NormSize,
                                   m_Data.m_Parameters.m_Alpha,
                                   m_Data.m_Parameters.m_Beta,
                                   m_Data.m_Parameters.m_K,
                                   m_Data.m_Parameters.m_DataLayout);
        }
        else
        {
            ARMNN_LOG(warning) << "Illegal NORMALIZATION mode in normalization_f32";
            return;
        }
    }
    else
    {
        ARMNN_LOG(warning) << "Lcr method (Jarret 2009: Local Contrast Normalization) not supported yet.";
        return;
    }
}

} // namespace armnn
