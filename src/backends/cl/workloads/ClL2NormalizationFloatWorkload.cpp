//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ClL2NormalizationFloatWorkload.hpp"
#include <cl/ClTensorHandle.hpp>
#include <backendsCommon/TensorHandle.hpp>
#include <aclCommon/ArmComputeUtils.hpp>

#include "ClWorkloadUtils.hpp"

namespace armnn
{
using namespace armcomputetensorutils;

arm_compute::Status ClL2NormalizationWorkloadValidate(const TensorInfo& input,
                                                      const TensorInfo& output,
                                                      const L2NormalizationDescriptor& descriptor)
{
    const arm_compute::TensorInfo aclInput  = BuildArmComputeTensorInfo(input, descriptor.m_DataLayout);
    const arm_compute::TensorInfo aclOutput = BuildArmComputeTensorInfo(output, descriptor.m_DataLayout);

    int axis = (descriptor.m_DataLayout == DataLayout::NCHW) ? 2 : 0;

    return arm_compute::CLL2NormalizeLayer::validate(&aclInput, &aclOutput, axis, descriptor.m_Eps);
}

ClL2NormalizationFloatWorkload::ClL2NormalizationFloatWorkload(const L2NormalizationQueueDescriptor& descriptor,
                                                               const WorkloadInfo& info,
                                                               const arm_compute::CLCompileContext& clCompileContext)
    : FloatWorkload<L2NormalizationQueueDescriptor>(descriptor, info)
{
    // Report Profiling Details
    ARMNN_REPORT_PROFILING_WORKLOAD_DESC("ClL2NormalizationFloatWorkload_Construct",
                                         descriptor.m_Parameters,
                                         info,
                                         this->GetGuid());

    m_Data.ValidateInputsOutputs("ClL2NormalizationFloatWorkload", 1, 1);

    arm_compute::ICLTensor& input  = static_cast<IClTensorHandle*>(m_Data.m_Inputs[0])->GetTensor();
    arm_compute::ICLTensor& output = static_cast<IClTensorHandle*>(m_Data.m_Outputs[0])->GetTensor();

    arm_compute::DataLayout aclDataLayout = ConvertDataLayout(m_Data.m_Parameters.m_DataLayout);
    input.info()->set_data_layout(aclDataLayout);
    output.info()->set_data_layout(aclDataLayout);

    int axis = (m_Data.m_Parameters.m_DataLayout == DataLayout::NCHW) ? 2 : 0;

    {
        ARMNN_SCOPED_PROFILING_EVENT(Compute::Undefined, "ClL2NormalizationFloatWorkload_configure");
        m_Layer.configure(clCompileContext, &input, &output, axis, m_Data.m_Parameters.m_Eps);
    }
}

void ClL2NormalizationFloatWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_CL_GUID("ClL2NormalizationFloatWorkload_Execute", this->GetGuid());
    RunClFunction(m_Layer, CHECK_LOCATION());
}

} //namespace armnn
