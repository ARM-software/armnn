//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ClPreluWorkload.hpp"
#include "ClWorkloadUtils.hpp"
#include <armnn/backends/TensorHandle.hpp>
#include <aclCommon/ArmComputeUtils.hpp>
#include <cl/ClLayerSupport.hpp>
#include <cl/ClTensorHandle.hpp>

namespace armnn
{

arm_compute::Status ClPreluWorkloadValidate(const TensorInfo& input,
                                            const TensorInfo& alpha,
                                            const TensorInfo& output)
{
    const arm_compute::TensorInfo aclInput = armcomputetensorutils::BuildArmComputeTensorInfo(input);
    const arm_compute::TensorInfo aclAlpha = armcomputetensorutils::BuildArmComputeTensorInfo(alpha);
    const arm_compute::TensorInfo aclOutput = armcomputetensorutils::BuildArmComputeTensorInfo(output);

    return arm_compute::CLPReluLayer::validate(&aclInput,
                                               &aclAlpha,
                                               &aclOutput);
}

ClPreluWorkload::ClPreluWorkload(const PreluQueueDescriptor& descriptor,
                                 const WorkloadInfo& info,
                                 const arm_compute::CLCompileContext& clCompileContext)
    : ClBaseWorkload<PreluQueueDescriptor>(descriptor, info)
{
    m_Data.ValidateInputsOutputs("ClPreluWorkload", 1, 1);

    arm_compute::ICLTensor& input = static_cast<IClTensorHandle*>(m_Data.m_Inputs[0])->GetTensor();
    arm_compute::ICLTensor& alpha = static_cast<IClTensorHandle*>(m_Data.m_Inputs[1])->GetTensor();
    arm_compute::ICLTensor& output = static_cast<IClTensorHandle*>(m_Data.m_Outputs[0])->GetTensor();

    {
        ARMNN_SCOPED_PROFILING_EVENT(Compute::Undefined, "ClPreluWorkload_configure");
        m_PreluLayer.configure(clCompileContext, &input, &alpha, &output);
    }
}

void ClPreluWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_CL_GUID("ClPreluWorkload_Execute", this->GetGuid());
    RunClFunction(m_PreluLayer, CHECK_LOCATION());
}

} //namespace armnn
