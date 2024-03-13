//
// Copyright © 2024 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ClScatterNdWorkload.hpp"

#include "ClWorkloadUtils.hpp"

#include <aclCommon/ArmComputeTensorUtils.hpp>
#include <cl/ClTensorHandle.hpp>

#include <arm_compute/function_info/ScatterInfo.h>

namespace armnn
{

using namespace armcomputetensorutils;

arm_compute::Status ClScatterNdWorkloadValidate(const TensorInfo& inputInfo,
                                                const TensorInfo& indicesInfo,
                                                const TensorInfo& updatesInfo,
                                                const TensorInfo& outputInfo,
                                                const ScatterNdDescriptor& descriptor)
{
    const arm_compute::TensorInfo aclInputInfo = BuildArmComputeTensorInfo(inputInfo);
    const arm_compute::TensorInfo aclIndicesInfo = BuildArmComputeTensorInfo(indicesInfo);
    const arm_compute::TensorInfo aclUpdatesInfo = BuildArmComputeTensorInfo(updatesInfo);
    const arm_compute::TensorInfo aclOutputInfo  = BuildArmComputeTensorInfo(outputInfo);

    arm_compute::ScatterInfo scatterInfo = BuildArmComputeScatterInfo(descriptor);

    return arm_compute::CLScatter::validate(descriptor.m_InputEnabled ? &aclInputInfo : nullptr,
                                            &aclUpdatesInfo,
                                            &aclIndicesInfo,
                                            &aclOutputInfo,
                                            scatterInfo);
}

ClScatterNdWorkload::ClScatterNdWorkload(const ScatterNdQueueDescriptor& descriptor,
                                         const WorkloadInfo& info,
                                         const arm_compute::CLCompileContext& clCompileContext)
    : ClBaseWorkload<ScatterNdQueueDescriptor>(descriptor, info)
{
    // Report Profiling Details
    ARMNN_REPORT_PROFILING_WORKLOAD_DESC("ClScatterNdWorkload_Construct",
                                         descriptor.m_Parameters,
                                         info,
                                         this->GetGuid());

    m_Data.ValidateInputsOutputs("ClScatterNdWorkload", 3, 1);

    arm_compute::ICLTensor& input   = static_cast<IClTensorHandle*>(m_Data.m_Inputs[0])->GetTensor();
    arm_compute::ICLTensor& updates = static_cast<IClTensorHandle*>(m_Data.m_Inputs[2])->GetTensor();
    arm_compute::ICLTensor& indices = static_cast<IClTensorHandle*>(m_Data.m_Inputs[1])->GetTensor();
    arm_compute::ICLTensor& output  = static_cast<IClTensorHandle*>(m_Data.m_Outputs[0])->GetTensor();

    arm_compute::ScatterInfo scatterInfo = BuildArmComputeScatterInfo(descriptor.m_Parameters);

    {
        ARMNN_SCOPED_PROFILING_EVENT_CL_NAME_GUID("ClScatterNdWorkload_configure");
        m_ScatterNdLayer.configure(clCompileContext,
                                   descriptor.m_Parameters.m_InputEnabled ? &input : nullptr,
                                   &updates,
                                   &indices,
                                   &output,
                                   scatterInfo);
    }
}

void ClScatterNdWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_CL_NAME_GUID("ClScatterNdWorkload_Execute");
    RunClFunction(m_ScatterNdLayer, CHECK_LOCATION());
}

} //namespace armnn
