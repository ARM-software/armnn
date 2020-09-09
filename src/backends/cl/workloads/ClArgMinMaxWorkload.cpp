//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ClArgMinMaxWorkload.hpp"
#include "ClWorkloadUtils.hpp"

#include <aclCommon/ArmComputeTensorUtils.hpp>

#include <backendsCommon/CpuTensorHandle.hpp>

#include <armnnUtils/TensorUtils.hpp>
#include <armnn/utility/NumericCast.hpp>

#include <cl/ClTensorHandle.hpp>
#include <cl/ClLayerSupport.hpp>

namespace
{
unsigned int CalcAclAxis(unsigned int numDimensions, unsigned int axisIndex)
{
    return (numDimensions - axisIndex) - 1;
}

} //namespace

namespace armnn
{

arm_compute::Status ClArgMinMaxWorkloadValidate(const TensorInfo& input,
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
        return arm_compute::CLArgMinMaxLayer::validate(&aclInput, aclAxis, &aclOutput,
                                                       arm_compute::ReductionOperation::ARG_IDX_MAX);
    }
    else
    {
        return arm_compute::CLArgMinMaxLayer::validate(&aclInput, aclAxis, &aclOutput,
                                                       arm_compute::ReductionOperation::ARG_IDX_MIN);
    }
}


ClArgMinMaxWorkload::ClArgMinMaxWorkload(const ArgMinMaxQueueDescriptor& descriptor,
                                         const WorkloadInfo& info)
        : BaseWorkload<ArgMinMaxQueueDescriptor>(descriptor, info)
{
    arm_compute::ICLTensor& input = static_cast<IClTensorHandle*>(this->m_Data.m_Inputs[0])->GetTensor();
    arm_compute::ICLTensor& output = static_cast<IClTensorHandle*>(this->m_Data.m_Outputs[0])->GetTensor();

    auto numDims = info.m_InputTensorInfos[0].GetNumDimensions();
    auto unsignedAxis = armnnUtils::GetUnsignedAxis(numDims, m_Data.m_Parameters.m_Axis);
    int aclAxis = armnn::numeric_cast<int>(CalcAclAxis(numDims, unsignedAxis));

    if (m_Data.m_Parameters.m_Function == ArgMinMaxFunction::Max)
    {
        m_ArgMinMaxLayer.configure(&input, aclAxis, &output, arm_compute::ReductionOperation::ARG_IDX_MAX);
    }
    else
    {
        m_ArgMinMaxLayer.configure(&input, aclAxis, &output, arm_compute::ReductionOperation::ARG_IDX_MIN);
    }
}

void ClArgMinMaxWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_CL("ClArgMinMaxWorkload_Execute");
    RunClFunction(m_ArgMinMaxLayer, CHECK_LOCATION());
}

} //namespace armnn

