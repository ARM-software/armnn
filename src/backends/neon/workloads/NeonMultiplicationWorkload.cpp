//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "NeonMultiplicationWorkload.hpp"

#include "NeonWorkloadUtils.hpp"

#include <arm_compute/runtime/NEON/functions/NEPixelWiseMultiplication.h>

namespace armnn
{

arm_compute::Status NeonMultiplicationWorkloadValidate(const TensorInfo& input0,
                                                       const TensorInfo& input1,
                                                       const TensorInfo& output)
{
    const arm_compute::TensorInfo aclInput1 = armcomputetensorutils::BuildArmComputeTensorInfo(input0);
    const arm_compute::TensorInfo aclInput2 = armcomputetensorutils::BuildArmComputeTensorInfo(input1);
    const arm_compute::TensorInfo aclOutput = armcomputetensorutils::BuildArmComputeTensorInfo(output);

    // At the time of writing, configure() will fail if a rounding policy other than TO_ZERO is supplied to it,
    // when providing a scale of 1.0 for F32 tensors, even though the provided rounding policy appears to be
    // ignored for F32 tensors.
    return arm_compute::NEPixelWiseMultiplication::validate(&aclInput1,
                                                            &aclInput2,
                                                            &aclOutput,
                                                            1.0f,
                                                            arm_compute::ConvertPolicy::SATURATE,
                                                            arm_compute::RoundingPolicy::TO_ZERO);
}

NeonMultiplicationWorkload::NeonMultiplicationWorkload(const MultiplicationQueueDescriptor& descriptor,
                                                       const WorkloadInfo& info)
    : BaseWorkload<MultiplicationQueueDescriptor>(descriptor, info)
{
    m_Data.ValidateInputsOutputs("NeonMultiplicationWorkload", 2, 1);

    arm_compute::ITensor& input1 = boost::polymorphic_downcast<IAclTensorHandle*>(m_Data.m_Inputs[0])->GetTensor();
    arm_compute::ITensor& input2 = boost::polymorphic_downcast<IAclTensorHandle*>(m_Data.m_Inputs[1])->GetTensor();
    arm_compute::ITensor& output = boost::polymorphic_downcast<IAclTensorHandle*>(m_Data.m_Outputs[0])->GetTensor();

    // At the time of writing, configure() will fail if a rounding policy other than TO_ZERO is supplied to it,
    // when providing a scale of 1.0 for F32 tensors, even though the provided rounding policy appears to be
    // ignored for F32 tensors.
    auto layer = std::make_unique<arm_compute::NEPixelWiseMultiplication>();
    layer->configure(&input1,
                     &input2,
                     &output,
                     1.0f,
                     arm_compute::ConvertPolicy::SATURATE,
                     arm_compute::RoundingPolicy::TO_ZERO);
    m_PixelWiseMultiplication.reset(layer.release());
}

void NeonMultiplicationWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_NEON("NeonMultiplicationWorkload_Execute");
    m_PixelWiseMultiplication->run();
}

} //namespace armnn
