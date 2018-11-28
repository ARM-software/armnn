//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "RefNormalizationFloat32Workload.hpp"

#include "RefWorkloadUtils.hpp"
#include "TensorBufferArrayView.hpp"

#include "Profiling.hpp"

#include <armnn/Tensor.hpp>

#include <boost/log/trivial.hpp>
#include <boost/numeric/conversion/cast.hpp>

using namespace armnnUtils;

namespace armnn
{

// Helper function to compute "Within" normalization using Krichevsky 2012: Local Brightness Normalization.
static void NormalizeWithinUingLbr(const float*       inputData,
                                   float*             outputData,
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

    int radius = boost::numeric_cast<int>(norm_size / 2u); /* Strong Assumption on rounding Mode */

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
                            int i = boost::numeric_cast<int>(w) + x;
                            int j = boost::numeric_cast<int>(h) + y;

                            if ((i < 0) || (i >= boost::numeric_cast<int>(cols)))
                            {
                                continue;
                            }

                            if ((j < 0) || (j >= boost::numeric_cast<int>(rows)))
                            {
                                continue;
                            }

                            float inval = inputData[n * cols * rows * depth +
                                                    c * cols * rows +
                                                    boost::numeric_cast<unsigned int>(j) * cols +
                                                    boost::numeric_cast<unsigned int>(i)];

                            accumulated_scale += inval*inval;
                        }
                    }
                    outputData[n * cols * rows * depth +
                               c * cols * rows +
                               h * cols +
                               w] = inputData[n * cols * rows * depth +
                                              c * cols * rows +
                                              h * cols +
                                              w] / (powf((kappa + (accumulated_scale * alpha)), beta));
                }
            }
        }
    }
}

// Helper function to compute "Across" normalization using Krichevsky 2012: Local Brightness Normalization.
void NormalizeAcrossUingLbr(const float*       inputData,
                            float*             outputData,
                            const TensorShape& tensorShape,
                            uint32_t           norm_size,
                            float              alpha,
                            float              beta,
                            float              kappa,
                            DataLayout         dataLayout)
{
    TensorBufferArrayView<const float> input(tensorShape,
                                             inputData,
                                             dataLayout);
    TensorBufferArrayView<float> output(tensorShape,
                                        outputData,
                                        dataLayout);

    DataLayoutIndexed dataLayoutIndexed(dataLayout);

    const unsigned int batchSize = tensorShape[0];
    const unsigned int depth     = tensorShape[dataLayoutIndexed.GetChannelsIndex()];
    const unsigned int rows      = tensorShape[dataLayoutIndexed.GetHeightIndex()];
    const unsigned int cols      = tensorShape[dataLayoutIndexed.GetWidthIndex()];

    int radius = boost::numeric_cast<int>(norm_size / 2u); /* Strong Assumption on rounding Mode */

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
                        int k = boost::numeric_cast<int>(c) + z;

                        if ((k < 0) || (k >= boost::numeric_cast<int>(depth)))
                        {
                            continue;
                        }

                        float inval = input.Get(n, boost::numeric_cast<unsigned int>(k), h, w);

                        accumulated_scale += inval * inval;
                    }

                    float scale = kappa + (accumulated_scale * alpha);
                    scale = powf(scale, -beta);

                    output.Get(n, c, h, w) = scale * input.Get(n, c, h, w);
                }
            }
        }
    }
}

void RefNormalizationFloat32Workload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT(Compute::CpuRef, "RefNormalizationFloat32Workload_Execute");

    const TensorInfo& inputInfo = GetTensorInfo(m_Data.m_Inputs[0]);

    float*       outputData = GetOutputTensorDataFloat(0, m_Data);
    const float* inputData = GetInputTensorDataFloat(0, m_Data);

    if (NormalizationAlgorithmMethod::LocalBrightness == m_Data.m_Parameters.m_NormMethodType)
    {
        if (NormalizationAlgorithmChannel::Within == m_Data.m_Parameters.m_NormChannelType)
        {
            NormalizeWithinUingLbr(inputData,
                                   outputData,
                                   inputInfo.GetShape(),
                                   m_Data.m_Parameters.m_NormSize,
                                   m_Data.m_Parameters.m_Alpha,
                                   m_Data.m_Parameters.m_Beta,
                                   m_Data.m_Parameters.m_K);
        }
        else if (NormalizationAlgorithmChannel::Across == m_Data.m_Parameters.m_NormChannelType)
        {
            NormalizeAcrossUingLbr(inputData,
                                   outputData,
                                   inputInfo.GetShape(),
                                   m_Data.m_Parameters.m_NormSize,
                                   m_Data.m_Parameters.m_Alpha,
                                   m_Data.m_Parameters.m_Beta,
                                   m_Data.m_Parameters.m_K,
                                   m_Data.m_Parameters.m_DataLayout);
        }
        else
        {
            BOOST_LOG_TRIVIAL(warning) << "Illegal NORMALIZATION mode in normalization_f32";
            return;
        }
    }
    else
    {
        BOOST_LOG_TRIVIAL(warning) << "Lcr method (Jarret 2009: Local Contrast Normalization) not supported yet.";
        return;
    }
}

} //namespace armnn
