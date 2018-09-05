//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ClBaseConstantWorkload.hpp"
#include "backends/ArmComputeTensorUtils.hpp"
#include "backends/ClTensorHandle.hpp"
#include "backends/CpuTensorHandle.hpp"
#include "Half.hpp"

namespace armnn
{

template class ClBaseConstantWorkload<DataType::Float16, DataType::Float32>;
template class ClBaseConstantWorkload<DataType::QuantisedAsymm8>;

template<armnn::DataType... dataTypes>
void ClBaseConstantWorkload<dataTypes...>::Execute() const
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
        arm_compute::DataType computeDataType = static_cast<ClTensorHandle*>(data.m_Outputs[0])->GetDataType();

        switch (computeDataType)
        {
            case arm_compute::DataType::F16:
            {
                CopyArmComputeClTensorData(data.m_LayerOutput->GetConstTensor<Half>(), output);
                break;
            }
            case arm_compute::DataType::F32:
            {
                CopyArmComputeClTensorData(data.m_LayerOutput->GetConstTensor<float>(), output);
                break;
            }
            case arm_compute::DataType::QASYMM8:
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