//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "NeonConstantWorkload.hpp"

#include <arm_compute/core/Types.h>
#include <Half.hpp>
#include <aclCommon/ArmComputeTensorUtils.hpp>
#include <neon/NeonTensorHandle.hpp>
#include <backendsCommon/CpuTensorHandle.hpp>
#include <backendsCommon/Workload.hpp>

#include <boost/cast.hpp>

namespace armnn
{

NeonConstantWorkload::NeonConstantWorkload(const ConstantQueueDescriptor& descriptor,
                                           const WorkloadInfo& info)
    : BaseWorkload<ConstantQueueDescriptor>(descriptor, info)
    , m_RanOnce(false)
{
}

void NeonConstantWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_NEON("NeonConstantWorkload_Execute");

    using namespace armcomputetensorutils;

    // The intermediate tensor held by the corresponding layer output handler can be initialised with the
    // given data on the first inference, then reused for subsequent inferences.
    // The initialisation cannot happen at workload construction time since the ACL kernel for the next layer
    // may not have been configured at the time.
    if (!m_RanOnce)
    {
        const ConstantQueueDescriptor& data = this->m_Data;

        BOOST_ASSERT(data.m_LayerOutput != nullptr);
        arm_compute::ITensor& output =
            boost::polymorphic_downcast<NeonTensorHandle*>(data.m_Outputs[0])->GetTensor();
        arm_compute::DataType computeDataType =
            boost::polymorphic_downcast<NeonTensorHandle*>(data.m_Outputs[0])->GetDataType();

        switch (computeDataType)
        {
            case arm_compute::DataType::F16:
            {
                CopyArmComputeITensorData(data.m_LayerOutput->GetConstTensor<Half>(), output);
                break;
            }
            case arm_compute::DataType::F32:
            {
                CopyArmComputeITensorData(data.m_LayerOutput->GetConstTensor<float>(), output);
                break;
            }
            case arm_compute::DataType::QASYMM8:
            {
                CopyArmComputeITensorData(data.m_LayerOutput->GetConstTensor<uint8_t>(), output);
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
