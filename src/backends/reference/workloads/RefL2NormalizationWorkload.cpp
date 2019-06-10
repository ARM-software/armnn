//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "RefL2NormalizationWorkload.hpp"

#include "RefWorkloadUtils.hpp"
#include "Decoders.hpp"
#include "Encoders.hpp"
#include "DataLayoutIndexed.hpp"


#include "Profiling.hpp"

#include <cmath>

using namespace armnnUtils;

namespace armnn
{
RefL2NormalizationWorkload::RefL2NormalizationWorkload(
            const L2NormalizationQueueDescriptor& descriptor,
            const WorkloadInfo& info)
            : BaseWorkload<L2NormalizationQueueDescriptor>(descriptor, info) {}

    void RefL2NormalizationWorkload::Execute() const
    {
        ARMNN_SCOPED_PROFILING_EVENT(Compute::CpuRef, "RefL2NormalizationWorkload_Execute");

        const TensorInfo& inputInfo = GetTensorInfo(m_Data.m_Inputs[0]);
        const TensorInfo& outputInfo = GetTensorInfo(m_Data.m_Outputs[0]);

        auto inputDecoder  = MakeDecoder<float>(inputInfo, m_Data.m_Inputs[0]->Map());
        auto outputEncoder = MakeEncoder<float>(outputInfo, m_Data.m_Outputs[0]->Map());

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
                            unsigned int inputIndex = dataLayout.GetIndex(inputInfo.GetShape(), n, d, h, w);

                            (*inputDecoder)[inputIndex];
                            const float value = inputDecoder->Get();
                            reduction += value * value;
                        }

                        unsigned int index = dataLayout.GetIndex(inputInfo.GetShape(), n, c, h, w);

                        const float scale = 1.0f / sqrtf(reduction);

                        (*inputDecoder)[index];
                        (*outputEncoder)[index];
                        outputEncoder->Set(inputDecoder->Get() * scale);
                    }
                }
            }
        }
    }

} //namespace armnn
