//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//

#include "ClMultiplicationFloat32Workload.hpp"
#include "backends/ClTensorHandle.hpp"
#include "backends/CpuTensorHandle.hpp"

namespace armnn
{

ClMultiplicationFloat32Workload::ClMultiplicationFloat32Workload(const MultiplicationQueueDescriptor& descriptor,
                                                                 const WorkloadInfo& info)
    : Float32Workload<MultiplicationQueueDescriptor>(descriptor, info)
{
    m_Data.ValidateInputsOutputs("ClMultiplicationFloat32Workload", 2, 1);

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

void ClMultiplicationFloat32Workload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT(Compute::GpuAcc, "ClMultiplicationFloat32Workload_Execute");

    // Execute the layer
    m_PixelWiseMultiplication.run();
}

} //namespace armnn
