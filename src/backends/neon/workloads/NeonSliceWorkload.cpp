//
// Copyright © 2019 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "NeonSliceWorkload.hpp"

#include "NeonWorkloadUtils.hpp"

#include <armnn/utility/PolymorphicDowncast.hpp>

#include <aclCommon/ArmComputeTensorUtils.hpp>

#include <neon/NeonTensorHandle.hpp>

#include <arm_compute/core/Error.h>

namespace armnn
{

arm_compute::Status NeonSliceWorkloadValidate(const TensorInfo& input,
                                              const TensorInfo& output,
                                              const SliceDescriptor& descriptor)
{
    const arm_compute::TensorInfo aclInputInfo = armcomputetensorutils::BuildArmComputeTensorInfo(input);
    const arm_compute::TensorInfo aclOutputInfo = armcomputetensorutils::BuildArmComputeTensorInfo(output);

    arm_compute::Coordinates starts;
    arm_compute::Coordinates ends;

    std::tie(starts, ends) = SetNeonSliceData(descriptor.m_Begin, descriptor.m_Size);

    return arm_compute::NESlice::validate(&aclInputInfo, &aclOutputInfo, starts, ends);
}

NeonSliceWorkload::NeonSliceWorkload(const SliceQueueDescriptor& descriptor,
                                     const WorkloadInfo& info)
        : NeonBaseWorkload<SliceQueueDescriptor>(descriptor, info)
{
    // Report Profiling Details
    ARMNN_REPORT_PROFILING_WORKLOAD_DESC("NeonSliceWorkload_Construct",
                                         descriptor.m_Parameters,
                                         info,
                                         this->GetGuid());


    m_Data.ValidateInputsOutputs("NeonSliceWorkload", 1, 1);

    arm_compute::ITensor& input = PolymorphicDowncast<IAclTensorHandle*>(m_Data.m_Inputs[0])->GetTensor();
    arm_compute::ITensor& output = PolymorphicDowncast<IAclTensorHandle*>(m_Data.m_Outputs[0])->GetTensor();

    arm_compute::Coordinates starts;
    arm_compute::Coordinates ends;

    std::tie(starts, ends) = SetNeonSliceData(m_Data.m_Parameters.m_Begin, m_Data.m_Parameters.m_Size);

    m_SliceFunction.configure(&input, &output, starts, ends);
}

void NeonSliceWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_NEON_GUID("NeonSliceWorkload_Execute", this->GetGuid());
    m_SliceFunction.run();
}

} // namespace armnn
