//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "RefL2NormalizationFloat32Workload.hpp"

#include "RefWorkloadUtils.hpp"
#include "TensorBufferArrayView.hpp"

#include "Profiling.hpp"

#include <cmath>

namespace armnn
{

void RefL2NormalizationFloat32Workload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT(Compute::CpuRef, "RefL2NormalizationFloat32Workload_Execute");

    const TensorInfo& inputInfo = GetTensorInfo(m_Data.m_Inputs[0]);
    const TensorInfo& outputInfo = GetTensorInfo(m_Data.m_Outputs[0]);

    TensorBufferArrayView<const float> input(inputInfo.GetShape(), GetInputTensorDataFloat(0, m_Data));
    TensorBufferArrayView<float> output(outputInfo.GetShape(), GetOutputTensorDataFloat(0, m_Data));

    const unsigned int batchSize = inputInfo.GetShape()[0];
    const unsigned int depth = inputInfo.GetShape()[1];
    const unsigned int rows = inputInfo.GetShape()[2];
    const unsigned int cols = inputInfo.GetShape()[3];

    for (unsigned int n = 0; n < batchSize; ++n)
    {
        for (unsigned int d = 0; d < depth; ++d)
        {
            for (unsigned int h = 0; h < rows; ++h)
            {
                for (unsigned int w = 0; w < cols; ++w)
                {
                    float reduction = 0.0;
                    for (unsigned int c = 0; c < depth; ++c)
                    {
                        const float value = input.Get(n, c, h, w);
                        reduction += value * value;
                    }

                    // Using std::max(reduction, epsilon) below would prevent against division by 0.
                    // However, at the time of writing:
                    // - This is not supported by the ACL functions used to implement L2Normalization in the CL
                    //   backend.
                    // - The reference semantics for this operator do not include this parameter.
                    const float scale = 1.0f / sqrtf(reduction);
                    output.Get(n, d, h, w) = input.Get(n, d, h, w) * scale;
                }
            }
        }
    }
}

} //namespace armnn
