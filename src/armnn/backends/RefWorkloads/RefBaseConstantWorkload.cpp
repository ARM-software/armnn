//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "RefBaseConstantWorkload.hpp"

#include "RefWorkloadUtils.hpp"

#include <armnn/Types.hpp>

#include <boost/assert.hpp>

#include <cstring>

namespace armnn
{

template <armnn::DataType DataType>
void RefBaseConstantWorkload<DataType>::Execute() const
{
    // Considering the reference backend independently, it could be possible to initialise the intermediate tensor
    // created by the layer output handler at workload construction time, rather than at workload execution time.
    // However, this is not an option for other backends (e.g. CL). For consistency, we prefer to align all
    // implementations.
    // A similar argument can be made about performing the memory copy in the first place (the layer output handler
    // could have a non-owning reference to the layer output tensor managed by the const input layer); again, this is
    // not an option for other backends, and the extra complexity required to make this work for the reference backend
    // may not be worth the effort (skipping a memory copy in the first inference).
    if (!m_RanOnce)
    {
        const ConstantQueueDescriptor& data = this->m_Data;

        BOOST_ASSERT(data.m_LayerOutput != nullptr);

        const TensorInfo& outputInfo = GetTensorInfo(data.m_Outputs[0]);
        BOOST_ASSERT(data.m_LayerOutput->GetTensorInfo().GetNumBytes() == outputInfo.GetNumBytes());

        memcpy(GetOutputTensorData<void>(0, data), data.m_LayerOutput->GetConstTensor<void>(),
            outputInfo.GetNumBytes());

        m_RanOnce = true;
    }
}

template class RefBaseConstantWorkload<DataType::Float32>;
template class RefBaseConstantWorkload<DataType::QuantisedAsymm8>;

} //namespace armnn
