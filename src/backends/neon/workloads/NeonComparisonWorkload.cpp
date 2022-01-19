//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "NeonComparisonWorkload.hpp"
#include <aclCommon/ArmComputeUtils.hpp>
#include <aclCommon/ArmComputeTensorUtils.hpp>
#include <armnn/utility/PolymorphicDowncast.hpp>
#include <armnn/backends/TensorHandle.hpp>

namespace armnn
{
using namespace armcomputetensorutils;

arm_compute::Status NeonComparisonWorkloadValidate(const TensorInfo& input0,
                                                   const TensorInfo& input1,
                                                   const TensorInfo& output,
                                                   const ComparisonDescriptor& descriptor)
{
    const arm_compute::TensorInfo aclInput0 = BuildArmComputeTensorInfo(input0);
    const arm_compute::TensorInfo aclInput1 = BuildArmComputeTensorInfo(input1);
    const arm_compute::TensorInfo aclOutput = BuildArmComputeTensorInfo(output);

    const arm_compute::ComparisonOperation comparisonOperation = ConvertComparisonOperationToAcl(descriptor);

    const arm_compute::Status aclStatus = arm_compute::NEElementwiseComparison::validate(&aclInput0,
                                                                                         &aclInput1,
                                                                                         &aclOutput,
                                                                                         comparisonOperation);
    return aclStatus;
}

NeonComparisonWorkload::NeonComparisonWorkload(const ComparisonQueueDescriptor& descriptor, const WorkloadInfo& info)
    : NeonBaseWorkload<ComparisonQueueDescriptor>(descriptor, info)
{
    // Report Profiling Details
    ARMNN_REPORT_PROFILING_WORKLOAD_DESC("NeonComparisonWorkload_Construct",
                                         descriptor.m_Parameters,
                                         info,
                                         this->GetGuid());

    m_Data.ValidateInputsOutputs("NeonComparisonWorkload", 2, 1);

    arm_compute::ITensor& input0 = PolymorphicDowncast<IAclTensorHandle*>(m_Data.m_Inputs[0])->GetTensor();
    arm_compute::ITensor& input1 = PolymorphicDowncast<IAclTensorHandle*>(m_Data.m_Inputs[1])->GetTensor();
    arm_compute::ITensor& output = PolymorphicDowncast<IAclTensorHandle*>(m_Data.m_Outputs[0])->GetTensor();

    const arm_compute::ComparisonOperation comparisonOperation = ConvertComparisonOperationToAcl(m_Data.m_Parameters);

    m_ComparisonLayer.configure(&input0, &input1, &output, comparisonOperation);
}

void NeonComparisonWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_NEON_GUID("NeonComparisonWorkload_Execute", this->GetGuid());
    m_ComparisonLayer.run();
}

} //namespace armnn