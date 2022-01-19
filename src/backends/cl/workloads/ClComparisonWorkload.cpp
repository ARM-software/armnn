//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ClComparisonWorkload.hpp"

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

arm_compute::Status ClComparisonWorkloadValidate(const TensorInfo& input0,
                                                 const TensorInfo& input1,
                                                 const TensorInfo& output,
                                                 const ComparisonDescriptor& descriptor)
{
    const arm_compute::TensorInfo aclInput0Info = BuildArmComputeTensorInfo(input0);
    const arm_compute::TensorInfo aclInput1Info = BuildArmComputeTensorInfo(input1);
    const arm_compute::TensorInfo aclOutputInfo = BuildArmComputeTensorInfo(output);

    const arm_compute::ComparisonOperation comparisonOperation = ConvertComparisonOperationToAcl(descriptor);

    const arm_compute::Status aclStatus = arm_compute::CLComparison::validate(&aclInput0Info,
                                                                              &aclInput1Info,
                                                                              &aclOutputInfo,
                                                                              comparisonOperation);
    return aclStatus;
}

ClComparisonWorkload::ClComparisonWorkload(const ComparisonQueueDescriptor& descriptor,
                                           const WorkloadInfo& info,
                                           const arm_compute::CLCompileContext& clCompileContext)
    : ClBaseWorkload<ComparisonQueueDescriptor>(descriptor, info)
{
    // Report Profiling Details
    ARMNN_REPORT_PROFILING_WORKLOAD_DESC("NeonComparisonWorkload_Construct",
                                         descriptor.m_Parameters,
                                         info,
                                         this->GetGuid());

    m_Data.ValidateInputsOutputs("ClComparisonWorkload", 2, 1);

    arm_compute::ICLTensor& input0 = static_cast<IClTensorHandle*>(m_Data.m_Inputs[0])->GetTensor();
    arm_compute::ICLTensor& input1 = static_cast<IClTensorHandle*>(m_Data.m_Inputs[1])->GetTensor();
    arm_compute::ICLTensor& output = static_cast<IClTensorHandle*>(m_Data.m_Outputs[0])->GetTensor();

    const arm_compute::ComparisonOperation comparisonOperation = ConvertComparisonOperationToAcl(m_Data.m_Parameters);

    {
        ARMNN_SCOPED_PROFILING_EVENT(Compute::Undefined, "ClComparisonWorkload_configure");
        m_ComparisonLayer.configure(clCompileContext, &input0, &input1, &output, comparisonOperation);
    }
}

void ClComparisonWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_CL_GUID("ClComparisonWorkload_Execute", this->GetGuid());
    RunClFunction(m_ComparisonLayer, CHECK_LOCATION());
}

} //namespace armnn
