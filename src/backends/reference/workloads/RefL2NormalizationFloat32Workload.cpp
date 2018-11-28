//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "RefL2NormalizationFloat32Workload.hpp"

#include "RefWorkloadUtils.hpp"
#include "TensorBufferArrayView.hpp"

#include "Profiling.hpp"

#include <cmath>

using namespace armnnUtils;

namespace armnn
{

void RefL2NormalizationFloat32Workload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT(Compute::CpuRef, "RefL2NormalizationFloat32Workload_Execute");

    const TensorInfo& inputInfo = GetTensorInfo(m_Data.m_Inputs[0]);
    const TensorInfo& outputInfo = GetTensorInfo(m_Data.m_Outputs[0]);

    TensorBufferArrayView<const float> input(inputInfo.GetShape(),
                                             GetInputTensorDataFloat(0, m_Data),
                                             m_Data.m_Parameters.m_DataLayout);
    TensorBufferArrayView<float> output(outputInfo.GetShape(),
                                        GetOutputTensorDataFloat(0, m_Data),
                                        m_Data.m_Parameters.m_DataLayout);

    DataLayoutIndexed dataLayout(m_Data.m_Parameters.m_DataLayout);

    const unsigned int batches  = inputInfo.GetShape()[0];
    const unsigned int channels = inputInfo.GetShape()[dataLayout.GetChannelsIndex()];
    const unsigned int height   = inputInfo.GetShape()[dataLayout.GetHeightIndex()];
    const unsigned int width    = inputInfo.GetShape()[dataLayout.GetWidthIndex()];

    for (unsigned int n = 0; n < batches; ++n)
    {
        for (unsigned int c = 0; c < channels; ++c)
        {
            for (unsigned int h = 0; h < height; ++h)
            {
                for (unsigned int w = 0; w < width; ++w)
                {
                    float reduction = 0.0;
                    for (unsigned int d = 0; d < channels; ++d)
                    {
                        const float value = input.Get(n, d, h, w);
                        reduction += value * value;
                    }

                    // Using std::max(reduction, epsilon) below would prevent against division by 0.
                    // However, at the time of writing:
                    // - This is not supported by the ACL functions used to implement L2Normalization in the CL
                    //   backend.
                    // - The reference semantics for this operator do not include this parameter.
                    const float scale = 1.0f / sqrtf(reduction);
                    output.Get(n, c, h, w) = input.Get(n, c, h, w) * scale;
                }
            }
        }
    }
}

} //namespace armnn
