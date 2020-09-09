//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ClSpaceToDepthWorkload.hpp"
#include "ClWorkloadUtils.hpp"

#include <aclCommon/ArmComputeUtils.hpp>
#include <aclCommon/ArmComputeTensorUtils.hpp>
#include <backendsCommon/CpuTensorHandle.hpp>
#include <cl/ClTensorHandle.hpp>

#include <armnn/utility/NumericCast.hpp>

namespace armnn
{
using namespace armcomputetensorutils;

ClSpaceToDepthWorkload::ClSpaceToDepthWorkload(const SpaceToDepthQueueDescriptor& desc,
                                               const WorkloadInfo& info)
    : BaseWorkload<SpaceToDepthQueueDescriptor>(desc, info)
{
    m_Data.ValidateInputsOutputs("ClSpaceToDepthWorkload", 1, 1);

    arm_compute::DataLayout aclDataLayout = ConvertDataLayout(m_Data.m_Parameters.m_DataLayout);

    arm_compute::ICLTensor& input = static_cast<IClTensorHandle*>(m_Data.m_Inputs[0])->GetTensor();
    input.info()->set_data_layout(aclDataLayout);

    int32_t blockSize = armnn::numeric_cast<int32_t>(desc.m_Parameters.m_BlockSize);

    arm_compute::ICLTensor& output = static_cast<IClTensorHandle*>(m_Data.m_Outputs[0])->GetTensor();
    output.info()->set_data_layout(aclDataLayout);

    m_Layer.configure(&input, &output, blockSize);
}

void ClSpaceToDepthWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_CL("ClSpaceToDepthWorkload_Execute");
    RunClFunction(m_Layer, CHECK_LOCATION());
}

arm_compute::Status ClSpaceToDepthWorkloadValidate(const TensorInfo& input,
                                                   const TensorInfo& output,
                                                   const SpaceToDepthDescriptor& desc)
{
    DataLayout dataLayout = desc.m_DataLayout;
    const arm_compute::TensorInfo aclInputInfo = BuildArmComputeTensorInfo(input, dataLayout);

    int32_t blockSize = armnn::numeric_cast<int32_t>(desc.m_BlockSize);

    const arm_compute::TensorInfo aclOutputInfo = BuildArmComputeTensorInfo(output, dataLayout);

    const arm_compute::Status aclStatus = arm_compute::CLSpaceToDepthLayer::validate(&aclInputInfo,
                                                                                     &aclOutputInfo,
                                                                                     blockSize);
    return aclStatus;
}

} //namespace armnn
