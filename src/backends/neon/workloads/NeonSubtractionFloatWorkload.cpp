//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "NeonSubtractionFloatWorkload.hpp"
#include <aclCommon/ArmComputeTensorUtils.hpp>
#include <backendsCommon/CpuTensorHandle.hpp>

namespace armnn
{

arm_compute::Status NeonSubtractionWorkloadValidate(const TensorInfo& input0,
                                                    const TensorInfo& input1,
                                                    const TensorInfo& output)
{
    const arm_compute::TensorInfo aclInput0 = armcomputetensorutils::BuildArmComputeTensorInfo(input0);
    const arm_compute::TensorInfo aclInput1 = armcomputetensorutils::BuildArmComputeTensorInfo(input1);
    const arm_compute::TensorInfo aclOutput = armcomputetensorutils::BuildArmComputeTensorInfo(output);

    return arm_compute::NEArithmeticSubtraction::validate(&aclInput0,
                                                          &aclInput1,
                                                          &aclOutput,
                                                          arm_compute::ConvertPolicy::SATURATE);
}

NeonSubtractionFloatWorkload::NeonSubtractionFloatWorkload(const SubtractionQueueDescriptor& descriptor,
                                                           const WorkloadInfo& info)
    : FloatWorkload<SubtractionQueueDescriptor>(descriptor, info)
{
    m_Data.ValidateInputsOutputs("NeonSubtractionFloatWorkload", 2, 1);

    arm_compute::ITensor& input1 = boost::polymorphic_downcast<INeonTensorHandle*>(m_Data.m_Inputs[0])->GetTensor();
    arm_compute::ITensor& input2 = boost::polymorphic_downcast<INeonTensorHandle*>(m_Data.m_Inputs[1])->GetTensor();
    arm_compute::ITensor& output = boost::polymorphic_downcast<INeonTensorHandle*>(m_Data.m_Outputs[0])->GetTensor();

    m_SubLayer.configure(&input1, &input2, &output, arm_compute::ConvertPolicy::SATURATE);
}

void NeonSubtractionFloatWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_NEON("NeonSubtractionFloatWorkload_Execute");
    m_SubLayer.run();
}

} //namespace armnn
