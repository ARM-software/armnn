//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ClBatchToSpaceNdWorkload.hpp"

#include <cl/ClTensorHandle.hpp>
#include <armnn/backends/TensorHandle.hpp>
#include <aclCommon/ArmComputeTensorUtils.hpp>

#include <armnn/utility/NumericCast.hpp>

#include "ClWorkloadUtils.hpp"

namespace armnn
{
using namespace armcomputetensorutils;

ClBatchToSpaceNdWorkload::ClBatchToSpaceNdWorkload(const BatchToSpaceNdQueueDescriptor& descriptor,
                                                   const WorkloadInfo& info,
                                                   const arm_compute::CLCompileContext& clCompileContext)
    : ClBaseWorkload<BatchToSpaceNdQueueDescriptor>(descriptor, info)
{
    // Report Profiling Details
    ARMNN_REPORT_PROFILING_WORKLOAD_DESC("ClBatchToSpaceWorkload_Construct",
                                         descriptor.m_Parameters,
                                         info,
                                         this->GetGuid());

    m_Data.ValidateInputsOutputs("ClBatchToSpaceNdWorkload", 1, 1);

    arm_compute::DataLayout aclDataLayout = ConvertDataLayout(m_Data.m_Parameters.m_DataLayout);

    arm_compute::ICLTensor& input = static_cast<IClTensorHandle*>(m_Data.m_Inputs[0])->GetTensor();
    input.info()->set_data_layout(aclDataLayout);

    // ArmNN blockShape is [H, W] Cl asks for W, H
    int32_t blockHeight = armnn::numeric_cast<int32_t>(descriptor.m_Parameters.m_BlockShape[0]);
    int32_t blockWidth = armnn::numeric_cast<int32_t>(descriptor.m_Parameters.m_BlockShape[1]);

    arm_compute::ICLTensor& output = static_cast<IClTensorHandle*>(m_Data.m_Outputs[0])->GetTensor();
    output.info()->set_data_layout(aclDataLayout);

    {
        ARMNN_SCOPED_PROFILING_EVENT(Compute::Undefined, "ClBatchToSpaceNdWorkload_configure");
        m_Layer.configure(clCompileContext, &input, blockWidth, blockHeight, &output);
    }
}

void ClBatchToSpaceNdWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_CL_GUID("ClBatchToSpaceNdWorkload_Execute", this->GetGuid());
    RunClFunction(m_Layer, CHECK_LOCATION());
}

arm_compute::Status ClBatchToSpaceNdWorkloadValidate(const TensorInfo& input,
                                                     const TensorInfo& output,
                                                     const BatchToSpaceNdDescriptor& descriptor)
{
    DataLayout dataLayout = descriptor.m_DataLayout;
    const arm_compute::TensorInfo aclInputInfo = BuildArmComputeTensorInfo(input, dataLayout);

    // ArmNN blockShape is [H, W] Cl asks for W, H
    int32_t blockHeight = armnn::numeric_cast<int32_t>(descriptor.m_BlockShape[0]);
    int32_t blockWidth = armnn::numeric_cast<int32_t>(descriptor.m_BlockShape[1]);

    const arm_compute::TensorInfo aclOutputInfo = BuildArmComputeTensorInfo(output, dataLayout);

    const arm_compute::Status aclStatus = arm_compute::CLBatchToSpaceLayer::validate(&aclInputInfo,
                                                                                     blockWidth,
                                                                                     blockHeight,
                                                                                     &aclOutputInfo);
    return aclStatus;
}

} //namespace armnn
