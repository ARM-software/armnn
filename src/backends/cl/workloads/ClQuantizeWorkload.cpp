//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ClQuantizeWorkload.hpp"
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

arm_compute::Status ClQuantizeWorkloadValidate(const TensorInfo& input,
                                               const TensorInfo& output)
{
    const arm_compute::TensorInfo aclInputInfo  = BuildArmComputeTensorInfo(input);
    const arm_compute::TensorInfo aclOutputInfo = BuildArmComputeTensorInfo(output);

    return arm_compute::CLQuantizationLayer::validate(&aclInputInfo,
                                                      &aclOutputInfo);
}

ClQuantizeWorkload::ClQuantizeWorkload(const QuantizeQueueDescriptor& descriptor,
                                       const WorkloadInfo& info,
                                       const arm_compute::CLCompileContext& clCompileContext)
    : ClBaseWorkload<QuantizeQueueDescriptor>(descriptor, info)
{
    m_Data.ValidateInputsOutputs("ClQuantizeWorkload", 1, 1);

    arm_compute::ICLTensor& input  = static_cast<IClTensorHandle*>(m_Data.m_Inputs[0])->GetTensor();
    arm_compute::ICLTensor& output = static_cast<IClTensorHandle*>(m_Data.m_Outputs[0])->GetTensor();

    {
        ARMNN_SCOPED_PROFILING_EVENT(Compute::Undefined, "ClQuantizeWorkload_configure");
        m_Layer.configure(clCompileContext, &input, &output);
    }
}

void ClQuantizeWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_CL_GUID("ClQuantizeWorkload_Execute", this->GetGuid());
    RunClFunction(m_Layer, CHECK_LOCATION());
}

} //namespace armnn
