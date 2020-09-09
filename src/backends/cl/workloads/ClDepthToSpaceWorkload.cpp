//
// Copyright © 2019 Arm Ltd. All rights reserved.
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
                                                   const DepthToSpaceDescriptor& desc)
{
    DataLayout dataLayout = desc.m_DataLayout;
    const arm_compute::TensorInfo aclInputInfo = BuildArmComputeTensorInfo(input, dataLayout);

    int32_t blockSize = armnn::numeric_cast<int32_t>(desc.m_BlockSize);

    const arm_compute::TensorInfo aclOutputInfo = BuildArmComputeTensorInfo(output, dataLayout);

    const arm_compute::Status aclStatus = arm_compute::CLDepthToSpaceLayer::validate(&aclInputInfo,
                                                                                     &aclOutputInfo,
                                                                                     blockSize);
    return aclStatus;
}

ClDepthToSpaceWorkload::ClDepthToSpaceWorkload(const DepthToSpaceQueueDescriptor& desc,
                                               const WorkloadInfo& info)
    : BaseWorkload<DepthToSpaceQueueDescriptor>(desc, info)
{
    m_Data.ValidateInputsOutputs("ClDepthToSpaceWorkload", 1, 1);

    arm_compute::DataLayout aclDataLayout = ConvertDataLayout(m_Data.m_Parameters.m_DataLayout);

    arm_compute::ICLTensor& input =
        PolymorphicPointerDowncast<IClTensorHandle>(m_Data.m_Inputs[0])->GetTensor();
    input.info()->set_data_layout(aclDataLayout);

    int32_t blockSize = armnn::numeric_cast<int32_t>(desc.m_Parameters.m_BlockSize);

    arm_compute::ICLTensor& output =
        PolymorphicPointerDowncast<IClTensorHandle>(m_Data.m_Outputs[0])->GetTensor();
    output.info()->set_data_layout(aclDataLayout);

    m_Layer.configure(&input, &output, blockSize);
}

void ClDepthToSpaceWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_CL("ClDepthToSpaceWorkload_Execute");
    RunClFunction(m_Layer, CHECK_LOCATION());
}

} // namespace armnn
