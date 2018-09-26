//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ClPadWorkload.hpp"

#include <backends/cl/ClTensorHandle.hpp>
#include <backends/aclCommon/ArmComputeTensorUtils.hpp>
#include <arm_compute/core/Types.h>

#include "ClWorkloadUtils.hpp"

namespace armnn
{
using namespace armcomputetensorutils;

template <armnn::DataType... T>
ClPadWorkload<T...>::ClPadWorkload(const PadQueueDescriptor& descriptor, const WorkloadInfo& info)
: TypedWorkload<PadQueueDescriptor, T...>(descriptor, info)
{
    this->m_Data.ValidateInputsOutputs("ClPadWorkload", 1, 1);

    arm_compute::ICLTensor& input = static_cast<IClTensorHandle*>(this->m_Data.m_Inputs[0])->GetTensor();
    arm_compute::ICLTensor& output = static_cast<IClTensorHandle*>(this->m_Data.m_Outputs[0])->GetTensor();
    arm_compute::PaddingList padList = static_cast<arm_compute::PaddingList>(descriptor.m_Parameters.m_PadList);

    m_Layer.configure(&input, &output, padList);
}

template <armnn::DataType... T>
void ClPadWorkload<T...>::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_CL("ClPadWorkload_Execute");
    m_Layer.run();
}

bool ClPadValidate(const TensorInfo& input,
                   const TensorInfo& output,
                   const PadDescriptor& descriptor,
                   std::string* reasonIfUnsupported)
{
    const arm_compute::TensorInfo aclInputInfo = BuildArmComputeTensorInfo(input);
    const arm_compute::TensorInfo aclOutputInfo = BuildArmComputeTensorInfo(output);
    arm_compute::PaddingList padList = static_cast<arm_compute::PaddingList>(descriptor.m_PadList);

    const arm_compute::Status aclStatus = arm_compute::CLPadLayer::validate(&aclInputInfo,
                                                                            &aclOutputInfo,
                                                                            padList);

    const bool supported = (aclStatus.error_code() == arm_compute::ErrorCode::OK);
    if (!supported && reasonIfUnsupported)
    {
        *reasonIfUnsupported = aclStatus.error_description();
    }

    return supported;
}

} // namespace armnn

template class armnn::ClPadWorkload<armnn::DataType::Float16, armnn::DataType::Float32>;
template class armnn::ClPadWorkload<armnn::DataType::QuantisedAsymm8>;
