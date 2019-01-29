//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ClGreaterWorkload.hpp"

#include "ClWorkloadUtils.hpp"

#include <aclCommon/ArmComputeUtils.hpp>
#include <aclCommon/ArmComputeTensorUtils.hpp>

#include <backendsCommon/CpuTensorHandle.hpp>

#include <cl/ClLayerSupport.hpp>
#include <cl/ClTensorHandle.hpp>
#include <cl/ClLayerSupport.hpp>

namespace armnn
{

using namespace armcomputetensorutils;

arm_compute::Status ClGreaterWorkloadValidate(const TensorInfo& input0,
                                              const TensorInfo& input1,
                                              const TensorInfo& output)
{
    const arm_compute::TensorInfo aclInput0Info = BuildArmComputeTensorInfo(input0);
    const arm_compute::TensorInfo aclInput1Info = BuildArmComputeTensorInfo(input1);
    const arm_compute::TensorInfo aclOutputInfo = BuildArmComputeTensorInfo(output);

    const arm_compute::Status aclStatus = arm_compute::CLComparison::validate(
        &aclInput0Info,
        &aclInput1Info,
        &aclOutputInfo,
        arm_compute::ComparisonOperation::Greater);

    return aclStatus;
}

template<DataType T>
ClGreaterWorkload<T>::ClGreaterWorkload(const GreaterQueueDescriptor& descriptor,
                                        const WorkloadInfo& info)
    : MultiTypedWorkload<GreaterQueueDescriptor, T, DataType::Boolean>(descriptor, info)
{
    m_Data.ValidateInputsOutputs("ClGreaterWorkload", 2, 1);

    arm_compute::ICLTensor& input0 = static_cast<IClTensorHandle*>(m_Data.m_Inputs[0])->GetTensor();
    arm_compute::ICLTensor& input1 = static_cast<IClTensorHandle*>(m_Data.m_Inputs[1])->GetTensor();
    arm_compute::ICLTensor& output = static_cast<IClTensorHandle*>(m_Data.m_Outputs[0])->GetTensor();

    m_GreaterLayer.configure(&input0, &input1, &output, arm_compute::ComparisonOperation::Greater);
}

template<DataType T>
void ClGreaterWorkload<T>::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_CL("ClGreaterWorkload_Execute");
    RunClFunction(m_GreaterLayer, CHECK_LOCATION());
}

template class ClGreaterWorkload<DataType::Float32>;
template class ClGreaterWorkload<DataType::QuantisedAsymm8>;

} //namespace armnn
