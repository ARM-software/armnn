//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "NeonSoftmaxWorkload.hpp"
#include "NeonWorkloadUtils.hpp"

#include <armnn/utility/PolymorphicDowncast.hpp>

#include <aclCommon/ArmComputeUtils.hpp>
#include <aclCommon/ArmComputeTensorUtils.hpp>

#include <arm_compute/runtime/NEON/functions/NESoftmaxLayer.h>

namespace armnn
{

arm_compute::Status NeonSoftmaxWorkloadValidate(const TensorInfo& input,
                                                const TensorInfo& output,
                                                const SoftmaxDescriptor& descriptor)
{
    const arm_compute::TensorInfo aclInputInfo = armcomputetensorutils::BuildArmComputeTensorInfo(input);
    const arm_compute::TensorInfo aclOutputInfo = armcomputetensorutils::BuildArmComputeTensorInfo(output);

    int aclAxis = ComputeAclAxis(descriptor.m_Axis, input);
    return arm_compute::NESoftmaxLayer::validate(&aclInputInfo,
                                                 &aclOutputInfo,
                                                 descriptor.m_Beta,
                                                 aclAxis);
}

NeonSoftmaxWorkload::NeonSoftmaxWorkload(const SoftmaxQueueDescriptor& descriptor,
    const WorkloadInfo& info, std::shared_ptr<arm_compute::MemoryManagerOnDemand>& memoryManager)
    : NeonBaseWorkload<SoftmaxQueueDescriptor>(descriptor, info)
{
    // Report Profiling Details
    ARMNN_REPORT_PROFILING_WORKLOAD_DESC("NeonSoftmaxWorkload_Construct",
                                         descriptor.m_Parameters,
                                         info,
                                         this->GetGuid());

    m_Data.ValidateInputsOutputs("NeonSoftmaxWorkload", 1, 1);

    // The ArmCompute softmax layer uses 2D input/output tensors, so flatten the first three dimensions.
    arm_compute::ITensor& input = PolymorphicDowncast<IAclTensorHandle*>(m_Data.m_Inputs[0])->GetTensor();
    arm_compute::ITensor& output = PolymorphicDowncast<IAclTensorHandle*>(m_Data.m_Outputs[0])->GetTensor();

    auto layer = std::make_unique<arm_compute::NESoftmaxLayer>(memoryManager);
    int aclAxis = ComputeAclAxis(m_Data.m_Parameters.m_Axis, info.m_InputTensorInfos[0]);
    layer->configure(&input, &output, m_Data.m_Parameters.m_Beta, aclAxis);
    m_SoftmaxLayer.reset(layer.release());
}

void NeonSoftmaxWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_NEON_GUID("NeonSoftmaxWorkload_Execute", this->GetGuid());
    m_SoftmaxLayer->run();
}

} //namespace armnn

