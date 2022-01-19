//
// Copyright © 2019 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "NeonArgMinMaxWorkload.hpp"
#include "NeonWorkloadUtils.hpp"

#include <aclCommon/ArmComputeTensorUtils.hpp>

#include <armnn/backends/TensorHandle.hpp>

#include <armnn/utility/NumericCast.hpp>
#include <armnn/utility/PolymorphicDowncast.hpp>
#include <armnnUtils/TensorUtils.hpp>

#include <arm_compute/runtime/NEON/functions/NEArgMinMaxLayer.h>

namespace
{
unsigned int CalcAclAxis(unsigned int numDimensions, unsigned int axisIndex)
{
    return (numDimensions - axisIndex) - 1;
}

} //namespace

namespace armnn
{

arm_compute::Status NeonArgMinMaxWorkloadValidate(const TensorInfo& input,
                                                  const TensorInfo& output,
                                                  const ArgMinMaxDescriptor& descriptor)
{
    const arm_compute::TensorInfo aclInput = armcomputetensorutils::BuildArmComputeTensorInfo(input);
    const arm_compute::TensorInfo aclOutput = armcomputetensorutils::BuildArmComputeTensorInfo(output);

    auto numDims = input.GetNumDimensions();
    auto unsignedAxis = armnnUtils::GetUnsignedAxis(numDims, descriptor.m_Axis);
    int aclAxis = armnn::numeric_cast<int>(CalcAclAxis(numDims, unsignedAxis));

    if (descriptor.m_Function == ArgMinMaxFunction::Max)
    {
        return arm_compute::NEArgMinMaxLayer::validate(&aclInput, aclAxis, &aclOutput,
                                                       arm_compute::ReductionOperation::ARG_IDX_MAX);
    }
    else
    {
        return arm_compute::NEArgMinMaxLayer::validate(&aclInput, aclAxis, &aclOutput,
                                                       arm_compute::ReductionOperation::ARG_IDX_MIN);
    }
}


NeonArgMinMaxWorkload::NeonArgMinMaxWorkload(const ArgMinMaxQueueDescriptor& descriptor,
                                             const WorkloadInfo& info)
        : NeonBaseWorkload<ArgMinMaxQueueDescriptor>(descriptor, info)
{
    // Report Profiling Details
    ARMNN_REPORT_PROFILING_WORKLOAD_DESC("NeonArgMinMaxWorkload_Construct",
                                         descriptor.m_Parameters,
                                         info,
                                         this->GetGuid());

    arm_compute::ITensor& input = PolymorphicDowncast<IAclTensorHandle*>(m_Data.m_Inputs[0])->GetTensor();
    arm_compute::ITensor& output = PolymorphicDowncast<IAclTensorHandle*>(m_Data.m_Outputs[0])->GetTensor();

    auto numDims = info.m_InputTensorInfos[0].GetNumDimensions();
    auto unsignedAxis = armnnUtils::GetUnsignedAxis(numDims, m_Data.m_Parameters.m_Axis);
    int aclAxis = armnn::numeric_cast<int>(CalcAclAxis(numDims, unsignedAxis));

    auto layer = std::make_unique<arm_compute::NEArgMinMaxLayer>();

    if (m_Data.m_Parameters.m_Function == ArgMinMaxFunction::Max)
    {
        layer->configure(&input, aclAxis, &output, arm_compute::ReductionOperation::ARG_IDX_MAX);
    }
    else
    {
        layer->configure(&input, aclAxis, &output, arm_compute::ReductionOperation::ARG_IDX_MIN);
    }

    m_ArgMinMaxLayer.reset(layer.release());
}

void NeonArgMinMaxWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_NEON_GUID("NeonArgMinMaxWorkload_Execute", this->GetGuid());
    m_ArgMinMaxLayer->run();
}

} //namespace armnn

