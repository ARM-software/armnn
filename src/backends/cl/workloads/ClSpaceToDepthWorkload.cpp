//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ClSpaceToDepthWorkload.hpp"
#include "ClWorkloadUtils.hpp"

#include <aclCommon/ArmComputeUtils.hpp>
#include <aclCommon/ArmComputeTensorUtils.hpp>
#include <armnn/backends/TensorHandle.hpp>
#include <cl/ClTensorHandle.hpp>

#include <armnn/utility/NumericCast.hpp>

namespace armnn
{
using namespace armcomputetensorutils;

ClSpaceToDepthWorkload::ClSpaceToDepthWorkload(const SpaceToDepthQueueDescriptor& descriptor,
                                               const WorkloadInfo& info,
                                               const arm_compute::CLCompileContext& clCompileContext)
    : ClBaseWorkload<SpaceToDepthQueueDescriptor>(descriptor, info)
{
    // Report Profiling Details
    ARMNN_REPORT_PROFILING_WORKLOAD_DESC("ClSpaceToDepthWorkload_Construct",
                                         descriptor.m_Parameters,
                                         info,
                                         this->GetGuid());
    m_Data.ValidateInputsOutputs("ClSpaceToDepthWorkload", 1, 1);

    arm_compute::DataLayout aclDataLayout = ConvertDataLayout(m_Data.m_Parameters.m_DataLayout);

    arm_compute::ICLTensor& input = static_cast<IClTensorHandle*>(m_Data.m_Inputs[0])->GetTensor();
    input.info()->set_data_layout(aclDataLayout);

    int32_t blockSize = armnn::numeric_cast<int32_t>(descriptor.m_Parameters.m_BlockSize);

    arm_compute::ICLTensor& output = static_cast<IClTensorHandle*>(m_Data.m_Outputs[0])->GetTensor();
    output.info()->set_data_layout(aclDataLayout);

    {
        ARMNN_SCOPED_PROFILING_EVENT(Compute::Undefined, "ClSpaceToDepthWorkload_configure");
        m_Layer.configure(clCompileContext, &input, &output, blockSize);
    }
}

void ClSpaceToDepthWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_CL_GUID("ClSpaceToDepthWorkload_Execute", this->GetGuid());
    RunClFunction(m_Layer, CHECK_LOCATION());
}

arm_compute::Status ClSpaceToDepthWorkloadValidate(const TensorInfo& input,
                                                   const TensorInfo& output,
                                                   const SpaceToDepthDescriptor& descriptor)
{
    DataLayout dataLayout = descriptor.m_DataLayout;
    const arm_compute::TensorInfo aclInputInfo = BuildArmComputeTensorInfo(input, dataLayout);

    int32_t blockSize = armnn::numeric_cast<int32_t>(descriptor.m_BlockSize);

    const arm_compute::TensorInfo aclOutputInfo = BuildArmComputeTensorInfo(output, dataLayout);

    const arm_compute::Status aclStatus = arm_compute::CLSpaceToDepthLayer::validate(&aclInputInfo,
                                                                                     &aclOutputInfo,
                                                                                     blockSize);
    return aclStatus;
}

} //namespace armnn
