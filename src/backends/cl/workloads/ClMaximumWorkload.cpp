//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ClMaximumWorkload.hpp"

#include "ClWorkloadUtils.hpp"

#include <aclCommon/ArmComputeUtils.hpp>
#include <aclCommon/ArmComputeTensorUtils.hpp>

#include <armnn/backends/TensorHandle.hpp>

#include <cl/ClLayerSupport.hpp>
#include <cl/ClTensorHandle.hpp>
#include <cl/ClLayerSupport.hpp>

namespace armnn
{

using namespace armcomputetensorutils;

arm_compute::Status ClMaximumWorkloadValidate(const TensorInfo& input0,
                                              const TensorInfo& input1,
                                              const TensorInfo& output)
{
    const arm_compute::TensorInfo aclInput0Info = BuildArmComputeTensorInfo(input0);
    const arm_compute::TensorInfo aclInput1Info = BuildArmComputeTensorInfo(input1);
    const arm_compute::TensorInfo aclOutputInfo = BuildArmComputeTensorInfo(output);

    const arm_compute::Status aclStatus = arm_compute::CLElementwiseMax::validate(&aclInput0Info,
                                                                                  &aclInput1Info,
                                                                                  &aclOutputInfo);

    return aclStatus;
}

ClMaximumWorkload::ClMaximumWorkload(const MaximumQueueDescriptor& descriptor,
                                     const WorkloadInfo& info,
                                     const arm_compute::CLCompileContext& clCompileContext)
    : ClBaseWorkload<MaximumQueueDescriptor>(descriptor, info)
{
    m_Data.ValidateInputsOutputs("ClMaximumWorkload", 2, 1);

    arm_compute::ICLTensor& input0 = static_cast<IClTensorHandle*>(m_Data.m_Inputs[0])->GetTensor();
    arm_compute::ICLTensor& input1 = static_cast<IClTensorHandle*>(m_Data.m_Inputs[1])->GetTensor();
    arm_compute::ICLTensor& output = static_cast<IClTensorHandle*>(m_Data.m_Outputs[0])->GetTensor();

    {
        ARMNN_SCOPED_PROFILING_EVENT(Compute::Undefined, "ClMaximumWorkload_configure");
        m_MaximumLayer.configure(clCompileContext, &input0, &input1, &output);
    }
}

void ClMaximumWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_CL_GUID("ClMaximumWorkload_Execute", this->GetGuid());
    RunClFunction(m_MaximumLayer, CHECK_LOCATION());
}

} //namespace armnn
