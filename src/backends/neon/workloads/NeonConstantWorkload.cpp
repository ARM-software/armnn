//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "NeonConstantWorkload.hpp"

#include <arm_compute/core/Types.h>
#include <BFloat16.hpp>
#include <Half.hpp>
#include <aclCommon/ArmComputeTensorUtils.hpp>
#include <armnn/utility/PolymorphicDowncast.hpp>
#include <neon/NeonTensorHandle.hpp>
#include <armnn/backends/TensorHandle.hpp>
#include "NeonBaseWorkload.hpp"

namespace armnn
{

arm_compute::Status NeonConstantWorkloadValidate(const TensorInfo& output)
{
    const arm_compute::TensorInfo neonOutputInfo = armcomputetensorutils::BuildArmComputeTensorInfo(output);

    std::array<arm_compute::DataType,9> supportedTypes = {
            arm_compute::DataType::BFLOAT16,
            arm_compute::DataType::F16,
            arm_compute::DataType::F32,
            arm_compute::DataType::QASYMM8,
            arm_compute::DataType::QASYMM8_SIGNED,
            arm_compute::DataType::QSYMM16,
            arm_compute::DataType::QSYMM8,
            arm_compute::DataType::QSYMM8_PER_CHANNEL,
            arm_compute::DataType::S32
    };
    auto it = std::find(begin(supportedTypes), end(supportedTypes), neonOutputInfo.data_type());

    if (it != end(supportedTypes))
    {
        return arm_compute::Status{};
    }
    else
    {
        return arm_compute::Status{arm_compute::ErrorCode::RUNTIME_ERROR, "Unsupported DataType"};
    }
}

NeonConstantWorkload::NeonConstantWorkload(const ConstantQueueDescriptor& descriptor,
                                           const WorkloadInfo& info)
    : NeonBaseWorkload<ConstantQueueDescriptor>(descriptor, info)
    , m_RanOnce(false)
{
}

void NeonConstantWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_NEON_GUID("NeonConstantWorkload_Execute", this->GetGuid());

    using namespace armcomputetensorutils;

    // The intermediate tensor held by the corresponding layer output handler can be initialised with the
    // given data on the first inference, then reused for subsequent inferences.
    // The initialisation cannot happen at workload construction time since the ACL kernel for the next layer
    // may not have been configured at the time.
    if (!m_RanOnce)
    {
        const ConstantQueueDescriptor& data = this->m_Data;

        ARMNN_ASSERT(data.m_LayerOutput != nullptr);
        arm_compute::ITensor& output =
            PolymorphicDowncast<NeonTensorHandle*>(data.m_Outputs[0])->GetTensor();
        arm_compute::DataType computeDataType =
            PolymorphicDowncast<NeonTensorHandle*>(data.m_Outputs[0])->GetDataType();

        switch (computeDataType)
        {
            case arm_compute::DataType::BFLOAT16:
            {
                CopyArmComputeITensorData(data.m_LayerOutput->GetConstTensor<BFloat16>(), output);
                break;
            }
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
            case arm_compute::DataType::QASYMM8_SIGNED:
            {
                CopyArmComputeITensorData(data.m_LayerOutput->GetConstTensor<int8_t>(), output);
                break;
            }
            case arm_compute::DataType::QSYMM16:
            {
                CopyArmComputeITensorData(data.m_LayerOutput->GetConstTensor<int16_t>(), output);
                break;
            }
            case arm_compute::DataType::QSYMM8:
            case arm_compute::DataType::QSYMM8_PER_CHANNEL:
            {
                CopyArmComputeITensorData(data.m_LayerOutput->GetConstTensor<int8_t>(), output);
                break;
            }
            case arm_compute::DataType::S32:
            {
                CopyArmComputeITensorData(data.m_LayerOutput->GetConstTensor<int32_t>(), output);
                break;
            }
            default:
            {
                ARMNN_ASSERT_MSG(false, "Unknown data type");
                break;
            }
        }

        m_RanOnce = true;
    }
}

} //namespace armnn
