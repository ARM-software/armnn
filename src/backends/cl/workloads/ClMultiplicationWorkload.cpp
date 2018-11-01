//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ClMultiplicationWorkload.hpp"
#include <cl/ClTensorHandle.hpp>
#include <backendsCommon/CpuTensorHandle.hpp>
#include "ClWorkloadUtils.hpp"

namespace armnn
{

arm_compute::Status ClMultiplicationWorkloadValidate(const TensorInfo& input0,
                                                     const TensorInfo& input1,
                                                     const TensorInfo& output)
{
    const arm_compute::TensorInfo aclInput1 = armcomputetensorutils::BuildArmComputeTensorInfo(input0);
    const arm_compute::TensorInfo aclInput2 = armcomputetensorutils::BuildArmComputeTensorInfo(input1);
    const arm_compute::TensorInfo aclOutput = armcomputetensorutils::BuildArmComputeTensorInfo(output);

    // At the time of writing, configure() will fail if a rounding policy other than TO_ZERO is supplied to it,
    // when providing a scale of 1.0 for F32 tensors, even though the provided rounding policy appears to be
    // ignored for F32 tensors.
    return arm_compute::CLPixelWiseMultiplication::validate(&aclInput1,
                                                            &aclInput2,
                                                            &aclOutput,
                                                            1.0f,
                                                            arm_compute::ConvertPolicy::SATURATE,
                                                            arm_compute::RoundingPolicy::TO_ZERO);
}


ClMultiplicationWorkload::ClMultiplicationWorkload(const MultiplicationQueueDescriptor& descriptor,
                                                   const WorkloadInfo& info)
    : BaseWorkload<MultiplicationQueueDescriptor>(descriptor, info)
{
    m_Data.ValidateInputsOutputs("ClMultiplicationWorkload", 2, 1);

    arm_compute::ICLTensor& input0 = static_cast<IClTensorHandle*>(m_Data.m_Inputs[0])->GetTensor();
    arm_compute::ICLTensor& input1 = static_cast<IClTensorHandle*>(m_Data.m_Inputs[1])->GetTensor();
    arm_compute::ICLTensor& output = static_cast<IClTensorHandle*>(m_Data.m_Outputs[0])->GetTensor();
    // Construct
    m_PixelWiseMultiplication.configure(&input0,
                                        &input1,
                                        &output,
                                        1.0f,
                                        arm_compute::ConvertPolicy::SATURATE,
                                        arm_compute::RoundingPolicy::TO_NEAREST_EVEN);
}

void ClMultiplicationWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_CL("ClMultiplicationWorkload_Execute");
    RunClFunction(m_PixelWiseMultiplication, CHECK_LOCATION());
}

} //namespace armnn
