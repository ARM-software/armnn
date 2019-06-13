//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "NeonGreaterWorkload.hpp"
#include <aclCommon/ArmComputeTensorUtils.hpp>
#include <backendsCommon/CpuTensorHandle.hpp>

namespace armnn
{

arm_compute::Status NeonGreaterWorkloadValidate(const TensorInfo& input0,
                                                const TensorInfo& input1,
                                                const TensorInfo& output)
{
    const arm_compute::TensorInfo aclInput0 = armcomputetensorutils::BuildArmComputeTensorInfo(input0);
    const arm_compute::TensorInfo aclInput1 = armcomputetensorutils::BuildArmComputeTensorInfo(input1);
    const arm_compute::TensorInfo aclOutput = armcomputetensorutils::BuildArmComputeTensorInfo(output);

    return arm_compute::NEGreater::validate(&aclInput0,
                                            &aclInput1,
                                            &aclOutput);
}

template <DataType T>
NeonGreaterWorkload<T>::NeonGreaterWorkload(const GreaterQueueDescriptor& descriptor, const WorkloadInfo& info)
    : MultiTypedWorkload<GreaterQueueDescriptor, T, DataType::Boolean>(descriptor, info)
{
    m_Data.ValidateInputsOutputs("NeonGreaterWorkload", 2, 1);

    arm_compute::ITensor& input0 = boost::polymorphic_downcast<IAclTensorHandle*>(m_Data.m_Inputs[0])->GetTensor();
    arm_compute::ITensor& input1 = boost::polymorphic_downcast<IAclTensorHandle*>(m_Data.m_Inputs[1])->GetTensor();
    arm_compute::ITensor& output = boost::polymorphic_downcast<IAclTensorHandle*>(m_Data.m_Outputs[0])->GetTensor();

    m_GreaterLayer.configure(&input0, &input1, &output);
}

template <DataType T>
void NeonGreaterWorkload<T>::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_NEON("NeonGreaterWorkload_Execute");
    m_GreaterLayer.run();
}

template class NeonGreaterWorkload<DataType::Float32>;
template class NeonGreaterWorkload<DataType::QuantisedAsymm8>;

} //namespace armnn