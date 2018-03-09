//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//

#include "NeonMultiplicationFloat32Workload.hpp"


namespace armnn
{

NeonMultiplicationFloat32Workload::NeonMultiplicationFloat32Workload(const MultiplicationQueueDescriptor& descriptor,
                                                                     const WorkloadInfo& info)
    : Float32Workload<MultiplicationQueueDescriptor>(descriptor, info)
{
    m_Data.ValidateInputsOutputs("NeonMultiplicationFloat32Workload", 2, 1);

    arm_compute::ITensor& input1 = boost::polymorphic_downcast<INeonTensorHandle*>(m_Data.m_Inputs[0])->GetTensor();
    arm_compute::ITensor& input2 = boost::polymorphic_downcast<INeonTensorHandle*>(m_Data.m_Inputs[1])->GetTensor();
    arm_compute::ITensor& output = boost::polymorphic_downcast<INeonTensorHandle*>(m_Data.m_Outputs[0])->GetTensor();

    // At the time of writing, configure() will fail if a rounding policy other than TO_ZERO is supplied to it,
    // when providing a scale of 1.0 for F32 tensors, even though the provided rounding policy appears to be
    // ignored for F32 tensors.
    m_PixelWiseMultiplication.configure(&input1,
                                        &input2,
                                        &output,
                                        1.0f,
                                        arm_compute::ConvertPolicy::SATURATE,
                                        arm_compute::RoundingPolicy::TO_ZERO);
}

void NeonMultiplicationFloat32Workload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT(Compute::CpuAcc, "NeonMultiplicationFloat32Workload_Execute");
    m_PixelWiseMultiplication.run();
}

} //namespace armnn


