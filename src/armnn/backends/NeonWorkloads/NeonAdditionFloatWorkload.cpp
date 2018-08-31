//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//

#include "NeonAdditionFloatWorkload.hpp"
#include "backends/ArmComputeTensorUtils.hpp"
#include "backends/CpuTensorHandle.hpp"

namespace armnn
{

arm_compute::Status NeonAdditionWorkloadValidate(const TensorInfo& input0,
                                                 const TensorInfo& input1,
                                                 const TensorInfo& output)
{
    const arm_compute::TensorInfo aclInput0 = armcomputetensorutils::BuildArmComputeTensorInfo(input0);
    const arm_compute::TensorInfo aclInput1 = armcomputetensorutils::BuildArmComputeTensorInfo(input1);
    const arm_compute::TensorInfo aclOutput = armcomputetensorutils::BuildArmComputeTensorInfo(output);

    return arm_compute::NEArithmeticAddition::validate(&aclInput0,
                                                       &aclInput1,
                                                       &aclOutput,
                                                       arm_compute::ConvertPolicy::SATURATE);
}


NeonAdditionFloatWorkload::NeonAdditionFloatWorkload(const AdditionQueueDescriptor& descriptor,
                                                     const WorkloadInfo& info)
    : FloatWorkload<AdditionQueueDescriptor>(descriptor, info)
{
    m_Data.ValidateInputsOutputs("NeonAdditionFloatWorkload", 2, 1);

    arm_compute::ITensor& input1 = boost::polymorphic_downcast<INeonTensorHandle*>(m_Data.m_Inputs[0])->GetTensor();
    arm_compute::ITensor& input2 = boost::polymorphic_downcast<INeonTensorHandle*>(m_Data.m_Inputs[1])->GetTensor();
    arm_compute::ITensor& output = boost::polymorphic_downcast<INeonTensorHandle*>(m_Data.m_Outputs[0])->GetTensor();

    m_AddLayer.configure(&input1, &input2, &output, arm_compute::ConvertPolicy::SATURATE);
}

void NeonAdditionFloatWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_NEON("NeonAdditionFloatWorkload_Execute");
    m_AddLayer.run();
}

} //namespace armnn

