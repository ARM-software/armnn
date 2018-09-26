//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ClAdditionWorkload.hpp"

#include <backends/cl/ClTensorHandle.hpp>
#include <backends/CpuTensorHandle.hpp>
#include <backends/aclCommon/ArmComputeTensorUtils.hpp>

#include "ClWorkloadUtils.hpp"

namespace armnn
{
using namespace armcomputetensorutils;

static constexpr arm_compute::ConvertPolicy g_AclConvertPolicy = arm_compute::ConvertPolicy::SATURATE;

template <armnn::DataType... T>
ClAdditionWorkload<T...>::ClAdditionWorkload(const AdditionQueueDescriptor& descriptor,
                                                  const WorkloadInfo& info)
    : TypedWorkload<AdditionQueueDescriptor, T...>(descriptor, info)
{
    this->m_Data.ValidateInputsOutputs("ClAdditionWorkload", 2, 1);

    arm_compute::ICLTensor& input0 = static_cast<IClTensorHandle*>(this->m_Data.m_Inputs[0])->GetTensor();
    arm_compute::ICLTensor& input1 = static_cast<IClTensorHandle*>(this->m_Data.m_Inputs[1])->GetTensor();
    arm_compute::ICLTensor& output = static_cast<IClTensorHandle*>(this->m_Data.m_Outputs[0])->GetTensor();
    m_Layer.configure(&input0, &input1, &output, g_AclConvertPolicy);
}

template <armnn::DataType... T>
void ClAdditionWorkload<T...>::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_CL("ClAdditionWorkload_Execute");
    m_Layer.run();
}

bool ClAdditionValidate(const TensorInfo& input0,
                        const TensorInfo& input1,
                        const TensorInfo& output,
                        std::string* reasonIfUnsupported)
{
    const arm_compute::TensorInfo aclInput0Info = BuildArmComputeTensorInfo(input0);
    const arm_compute::TensorInfo aclInput1Info = BuildArmComputeTensorInfo(input1);
    const arm_compute::TensorInfo aclOutputInfo = BuildArmComputeTensorInfo(output);

    const arm_compute::Status aclStatus = arm_compute::CLArithmeticAddition::validate(&aclInput0Info,
                                                                                      &aclInput1Info,
                                                                                      &aclOutputInfo,
                                                                                      g_AclConvertPolicy);

    const bool supported = (aclStatus.error_code() == arm_compute::ErrorCode::OK);
    if (!supported && reasonIfUnsupported)
    {
        *reasonIfUnsupported = aclStatus.error_description();
    }

    return supported;
}

} //namespace armnn

template class armnn::ClAdditionWorkload<armnn::DataType::Float16, armnn::DataType::Float32>;
template class armnn::ClAdditionWorkload<armnn::DataType::QuantisedAsymm8>;
