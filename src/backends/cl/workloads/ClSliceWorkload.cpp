//
// Copyright © 2019 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ClSliceWorkload.hpp"

#include "ClWorkloadUtils.hpp"

#include <aclCommon/ArmComputeTensorUtils.hpp>
#include <armnn/utility/PolymorphicDowncast.hpp>

#include <cl/ClTensorHandle.hpp>

namespace armnn
{

arm_compute::Status ClSliceWorkloadValidate(const TensorInfo& input,
                                            const TensorInfo& output,
                                            const SliceDescriptor& descriptor)
{
    const arm_compute::TensorInfo aclInput  = armcomputetensorutils::BuildArmComputeTensorInfo(input);
    const arm_compute::TensorInfo aclOutput = armcomputetensorutils::BuildArmComputeTensorInfo(output);

    arm_compute::Coordinates starts;
    arm_compute::Coordinates ends;

    std::tie(starts, ends) = SetClSliceData(descriptor.m_Begin, descriptor.m_Size);

    return arm_compute::CLSlice::validate(&aclInput, &aclOutput, starts, ends);
}

ClSliceWorkload::ClSliceWorkload(const SliceQueueDescriptor& descriptor,
                                 const WorkloadInfo& info,
                                 const arm_compute::CLCompileContext& clCompileContext)
    : ClBaseWorkload<SliceQueueDescriptor>(descriptor, info)
{
    // Report Profiling Details
    ARMNN_REPORT_PROFILING_WORKLOAD_DESC("ClSliceWorkload_Construct",
                                         descriptor.m_Parameters,
                                         info,
                                         this->GetGuid());

    m_Data.ValidateInputsOutputs("ClSliceWorkload", 1, 1);

    arm_compute::ICLTensor& input  = PolymorphicDowncast<IClTensorHandle*>(m_Data.m_Inputs[0])->GetTensor();
    arm_compute::ICLTensor& output = PolymorphicDowncast<IClTensorHandle*>(m_Data.m_Outputs[0])->GetTensor();

    arm_compute::Coordinates starts;
    arm_compute::Coordinates ends;

    std::tie(starts, ends) = SetClSliceData(m_Data.m_Parameters.m_Begin, m_Data.m_Parameters.m_Size);

    {
        ARMNN_SCOPED_PROFILING_EVENT(Compute::Undefined, "ClSliceWorkload_configure");
        m_SliceFunction.configure(clCompileContext, &input, &output, starts, ends);
    }
}

void ClSliceWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_CL_GUID("ClSliceWorkload_Execute", this->GetGuid());
    RunClFunction(m_SliceFunction, CHECK_LOCATION());
}

} // namespace armnn
