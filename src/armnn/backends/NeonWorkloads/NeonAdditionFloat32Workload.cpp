//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//

#include "NeonAdditionFloat32Workload.hpp"
#include "backends/CpuTensorHandle.hpp"

namespace armnn
{

NeonAdditionFloat32Workload::NeonAdditionFloat32Workload(const AdditionQueueDescriptor& descriptor,
                                                         const WorkloadInfo& info)
    : Float32Workload<AdditionQueueDescriptor>(descriptor, info)
{
    m_Data.ValidateInputsOutputs("NeonAdditionFloat32Workload", 2, 1);

    arm_compute::ITensor& input1 = boost::polymorphic_downcast<INeonTensorHandle*>(m_Data.m_Inputs[0])->GetTensor();
    arm_compute::ITensor& input2 = boost::polymorphic_downcast<INeonTensorHandle*>(m_Data.m_Inputs[1])->GetTensor();
    arm_compute::ITensor& output = boost::polymorphic_downcast<INeonTensorHandle*>(m_Data.m_Outputs[0])->GetTensor();

    m_AddLayer.configure(&input1, &input2, &output, arm_compute::ConvertPolicy::SATURATE);
}

void NeonAdditionFloat32Workload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT(Compute::CpuAcc, "NeonAdditionFloat32Workload_Execute");
    m_AddLayer.run();
}

} //namespace armnn

