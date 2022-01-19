//
// Copyright © 2019 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ClDepthToSpaceWorkload.hpp"

#include "ClWorkloadUtils.hpp"

#include <aclCommon/ArmComputeTensorUtils.hpp>

#include <armnn/utility/NumericCast.hpp>
#include <armnn/utility/PolymorphicDowncast.hpp>

#include <cl/ClTensorHandle.hpp>

namespace armnn
{

using namespace armcomputetensorutils;

arm_compute::Status ClDepthToSpaceWorkloadValidate(const TensorInfo& input,
                                                   const TensorInfo& output,
                                                   const DepthToSpaceDescriptor& descriptor)
{
    DataLayout dataLayout = descriptor.m_DataLayout;
    const arm_compute::TensorInfo aclInputInfo = BuildArmComputeTensorInfo(input, dataLayout);

    int32_t blockSize = armnn::numeric_cast<int32_t>(descriptor.m_BlockSize);

    const arm_compute::TensorInfo aclOutputInfo = BuildArmComputeTensorInfo(output, dataLayout);

    const arm_compute::Status aclStatus = arm_compute::CLDepthToSpaceLayer::validate(&aclInputInfo,
                                                                                     &aclOutputInfo,
                                                                                     blockSize);
    return aclStatus;
}

ClDepthToSpaceWorkload::ClDepthToSpaceWorkload(const DepthToSpaceQueueDescriptor& descriptor,
                                               const WorkloadInfo& info,
                                               const arm_compute::CLCompileContext& clCompileContext)
    : ClBaseWorkload<DepthToSpaceQueueDescriptor>(descriptor, info)
{
    // Report Profiling Details
    ARMNN_REPORT_PROFILING_WORKLOAD_DESC("ClDepthToSpaceWorkload_Construct",
                                         descriptor.m_Parameters,
                                         info,
                                         this->GetGuid());

    m_Data.ValidateInputsOutputs("ClDepthToSpaceWorkload", 1, 1);

    arm_compute::DataLayout aclDataLayout = ConvertDataLayout(m_Data.m_Parameters.m_DataLayout);

    arm_compute::ICLTensor& input =
        PolymorphicPointerDowncast<IClTensorHandle>(m_Data.m_Inputs[0])->GetTensor();
    input.info()->set_data_layout(aclDataLayout);

    int32_t blockSize = armnn::numeric_cast<int32_t>(descriptor.m_Parameters.m_BlockSize);

    arm_compute::ICLTensor& output =
        PolymorphicPointerDowncast<IClTensorHandle>(m_Data.m_Outputs[0])->GetTensor();
    output.info()->set_data_layout(aclDataLayout);

    {
        ARMNN_SCOPED_PROFILING_EVENT(Compute::Undefined, "ClDepthToSpaceWorkload_configure");
        m_Layer.configure(clCompileContext, &input, &output, blockSize);
    }
}

void ClDepthToSpaceWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_CL_GUID("ClDepthToSpaceWorkload_Execute", this->GetGuid());
    RunClFunction(m_Layer, CHECK_LOCATION());
}

} // namespace armnn
