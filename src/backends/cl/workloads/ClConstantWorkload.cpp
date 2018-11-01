//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ClConstantWorkload.hpp"

#include <Half.hpp>
#include <aclCommon/ArmComputeTensorUtils.hpp>
#include <cl/ClTensorHandle.hpp>
#include <backendsCommon/CpuTensorHandle.hpp>

#include "ClWorkloadUtils.hpp"

namespace armnn
{

ClConstantWorkload::ClConstantWorkload(const ConstantQueueDescriptor& descriptor, const WorkloadInfo& info)
    : BaseWorkload<ConstantQueueDescriptor>(descriptor, info)
    , m_RanOnce(false)
{
}

void ClConstantWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_CL("ClConstantWorkload_Execute");

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
                CopyArmComputeClTensorData(output, data.m_LayerOutput->GetConstTensor<Half>());
                break;
            }
            case arm_compute::DataType::F32:
            {
                CopyArmComputeClTensorData(output, data.m_LayerOutput->GetConstTensor<float>());
                break;
            }
            case arm_compute::DataType::QASYMM8:
            {
                CopyArmComputeClTensorData(output, data.m_LayerOutput->GetConstTensor<uint8_t>());
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
