//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//

#include "ClBaseConstantWorkload.hpp"
#include "backends/ClTensorHandle.hpp"
#include "backends/CpuTensorHandle.hpp"

namespace armnn
{

template class ClBaseConstantWorkload<DataType::Float32>;
template class ClBaseConstantWorkload<DataType::QuantisedAsymm8>;

template<armnn::DataType dataType>
void ClBaseConstantWorkload<dataType>::Execute() const
{
    // The intermediate tensor held by the corresponding layer output handler can be initialised with the given data
    // on the first inference, then reused for subsequent inferences.
    // The initialisation cannot happen at workload construction time since the ACL kernel for the next layer may not
    // have been configured at the time.
    if (!m_RanOnce)
    {
        const ConstantQueueDescriptor& data = this->m_Data;

        BOOST_ASSERT(data.m_LayerOutput != nullptr);
        arm_compute::CLTensor& output = static_cast<ClTensorHandle*>(data.m_Outputs[0])->GetTensor();

        switch (dataType)
        {
            case DataType::Float32:
            {
                CopyArmComputeClTensorData(data.m_LayerOutput->GetConstTensor<float>(), output);
                break;
            }
            case DataType::QuantisedAsymm8:
            {
                CopyArmComputeClTensorData(data.m_LayerOutput->GetConstTensor<uint8_t>(), output);
                break;
            }
            default:
            {
                BOOST_ASSERT_MSG(false, "Unknown data type");
                break;
            }
        }

        m_RanOnce = true;
    }
}


} //namespace armnn