//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "NeonLogSoftmaxWorkload.hpp"
#include "NeonWorkloadUtils.hpp"

#include <armnn/utility/PolymorphicDowncast.hpp>

#include <aclCommon/ArmComputeUtils.hpp>
#include <aclCommon/ArmComputeTensorUtils.hpp>

#include <arm_compute/runtime/NEON/functions/NESoftmaxLayer.h>

namespace armnn
{

arm_compute::Status NeonLogSoftmaxWorkloadValidate(const TensorInfo& input,
                                                   const TensorInfo& output,
                                                   const LogSoftmaxDescriptor& descriptor)
{
    const arm_compute::TensorInfo aclInputInfo = armcomputetensorutils::BuildArmComputeTensorInfo(input);
    const arm_compute::TensorInfo aclOutputInfo = armcomputetensorutils::BuildArmComputeTensorInfo(output);

    int aclAxis = ComputeAclAxis(descriptor.m_Axis, input);
    return arm_compute::NELogSoftmaxLayer::validate(&aclInputInfo,
                                                    &aclOutputInfo,
                                                    descriptor.m_Beta,
                                                    aclAxis);
}

NeonLogSoftmaxWorkload::NeonLogSoftmaxWorkload(const LogSoftmaxQueueDescriptor& descriptor,
                                               const WorkloadInfo& info,
                                               std::shared_ptr<arm_compute::MemoryManagerOnDemand>& memoryManager)
    : NeonBaseWorkload<LogSoftmaxQueueDescriptor>(descriptor, info)
{
    // Report Profiling Details
    ARMNN_REPORT_PROFILING_WORKLOAD_DESC("NeonLogSoftmaxWorkload_Construct",
                                         descriptor.m_Parameters,
                                         info,
                                         this->GetGuid());

    m_Data.ValidateInputsOutputs("NeonLogSoftmaxWorkload", 1, 1);

    arm_compute::ITensor& input = PolymorphicDowncast<IAclTensorHandle*>(m_Data.m_Inputs[0])->GetTensor();
    arm_compute::ITensor& output = PolymorphicDowncast<IAclTensorHandle*>(m_Data.m_Outputs[0])->GetTensor();

    auto layer = std::make_unique<arm_compute::NELogSoftmaxLayer>(memoryManager);
    int aclAxis = ComputeAclAxis(m_Data.m_Parameters.m_Axis, info.m_InputTensorInfos[0]);
    layer->configure(&input, &output, m_Data.m_Parameters.m_Beta, aclAxis);
    m_LogSoftmaxLayer.reset(layer.release());
}

void NeonLogSoftmaxWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_NEON_GUID("NeonLogSoftmaxWorkload_Execute", this->GetGuid());
    m_LogSoftmaxLayer->run();
}

} //namespace armnn

