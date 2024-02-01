//
// Copyright © 2024 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "GpuFsaConstantWorkload.hpp"
#include "GpuFsaWorkloadUtils.hpp"

#include <Half.hpp>
#include <aclCommon/ArmComputeTensorUtils.hpp>
#include <gpuFsa/GpuFsaTensorHandle.hpp>
#include <armnn/backends/TensorHandle.hpp>

namespace armnn
{

arm_compute::Status GpuFsaConstantWorkloadValidate(const TensorInfo& output)
{
    const arm_compute::TensorInfo neonOutputInfo = armcomputetensorutils::BuildArmComputeTensorInfo(output);

    std::array<arm_compute::DataType,8> supportedTypes = {
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

GpuFsaConstantWorkload::GpuFsaConstantWorkload(const ConstantQueueDescriptor& descriptor,
                                             const WorkloadInfo& info,
                                             const arm_compute::CLCompileContext&)
        : GpuFsaBaseWorkload<ConstantQueueDescriptor>(descriptor, info)
        , m_RanOnce(false)
{
}

void GpuFsaConstantWorkload::Execute() const
{
    // The intermediate tensor held by the corresponding layer output handler can be initialised with the given data
    // on the first inference, then reused for subsequent inferences.
    // The initialisation cannot happen at workload construction time since the ACL kernel for the next layer may not
    // have been configured at the time.
    if (!m_RanOnce)
    {
        const ConstantQueueDescriptor& data = this->m_Data;
        arm_compute::CLTensor& output = static_cast<GpuFsaTensorHandle*>(data.m_Outputs[0])->GetTensor();
        arm_compute::DataType computeDataType = static_cast<GpuFsaTensorHandle*>(data.m_Outputs[0])->GetDataType();

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
            case arm_compute::DataType::QASYMM8_SIGNED:
            {
                CopyArmComputeClTensorData(output, data.m_LayerOutput->GetConstTensor<int8_t>());
                break;
            }
            case arm_compute::DataType::QSYMM16:
            {
                CopyArmComputeClTensorData(output, data.m_LayerOutput->GetConstTensor<int16_t>());
                break;
            }
            case arm_compute::DataType::QSYMM8:
            case arm_compute::DataType::QSYMM8_PER_CHANNEL:
            {
                CopyArmComputeClTensorData(output, data.m_LayerOutput->GetConstTensor<int8_t>());
                break;
            }
            case arm_compute::DataType::S32:
            {
                CopyArmComputeClTensorData(output, data.m_LayerOutput->GetConstTensor<int32_t>());
                break;
            }
            default:
            {
                throw InvalidArgumentException("Unknown data type passed to GpuFsaConstantWorkload::Execute()");
                break;
            }
        }

        m_RanOnce = true;
    }
}

} //namespace armnn
