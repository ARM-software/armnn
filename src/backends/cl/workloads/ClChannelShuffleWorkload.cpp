//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ClChannelShuffleWorkload.hpp"
#include "ClWorkloadUtils.hpp"

#include <armnn/utility/PolymorphicDowncast.hpp>

#include <aclCommon/ArmComputeTensorUtils.hpp>

#include <cl/ClTensorHandle.hpp>

using namespace armnn::armcomputetensorutils;

namespace armnn
{

arm_compute::Status ClChannelShuffleValidate(const TensorInfo& input,
                                             const TensorInfo& output,
                                             const ChannelShuffleDescriptor& descriptor)
{
    arm_compute::TensorInfo aclInputInfo  = BuildArmComputeTensorInfo(input);
    arm_compute::TensorInfo aclOutputInfo = BuildArmComputeTensorInfo(output);

    // In Arm NN and in NNAPI, channel shuffle implementation is datalayout agnostic and it has axis as a parameter.
    // The channel shuffle Implementation for Neon is dependent on datalayout and does not have axis as a parameter,
    // it only supports channel shuffle for 4D tensors in dimension C (1 or 3).
    arm_compute::DataLayout aclDataLayout;
    if (input.GetNumDimensions() == 4)
    {
        switch (descriptor.m_Axis)
        {
            case 1:
                aclDataLayout = ConvertDataLayout(armnn::DataLayout::NCHW);
                break;
            case 3:
                aclDataLayout = ConvertDataLayout(armnn::DataLayout::NHWC);
                break;
            default:
                return arm_compute::Status{arm_compute::ErrorCode::RUNTIME_ERROR, "Unsupported axis"};
        }
        aclInputInfo.set_data_layout(aclDataLayout);
        aclOutputInfo.set_data_layout(aclDataLayout);
        return arm_compute::CLChannelShuffleLayer::validate(&aclInputInfo, &aclOutputInfo, descriptor.m_NumGroups);
    }
    else
    {
        return arm_compute::Status{arm_compute::ErrorCode::RUNTIME_ERROR, "Unsupported number of dimensions"};
    }
}

ClChannelShuffleWorkload::ClChannelShuffleWorkload(const ChannelShuffleQueueDescriptor& descriptor,
                                                   const WorkloadInfo& info,
                                                   const arm_compute::CLCompileContext& clCompileContext)
    : ClBaseWorkload<ChannelShuffleQueueDescriptor>(descriptor, info)
{
    // Report Profiling Details
    ARMNN_REPORT_PROFILING_WORKLOAD_DESC("ClChannelShufflenWorkload_Construct",
                                         descriptor.m_Parameters,
                                         info,
                                         this->GetGuid());

    m_Data.ValidateInputsOutputs("ClChannelShuffleWorkload", 1, 1);

    arm_compute::ICLTensor& input  = PolymorphicDowncast<ClTensorHandle*>(m_Data.m_Inputs[0])->GetTensor();
    arm_compute::ICLTensor& output = PolymorphicDowncast<ClTensorHandle*>(m_Data.m_Outputs[0])->GetTensor();

    // In Arm NN and in NNAPI, channel shuffle implementation is datalayout agnostic and it has axis as a parameter.
    // The channel shuffle Implementation for Neon is dependent on datalayout and does not have axis as a parameter,
    // it only supports channel shuffle for 4D tensors in dimension C (1 or 3).
    arm_compute::DataLayout aclDataLayout;
    switch (descriptor.m_Parameters.m_Axis)
    {
        case 1:
            aclDataLayout = ConvertDataLayout(armnn::DataLayout::NCHW);
            break;
        case 3:
            aclDataLayout = ConvertDataLayout(armnn::DataLayout::NHWC);
            break;
        default:
            ARMNN_ASSERT_MSG(false, "Unsupported axis");
            break;
    }
    input.info()->set_data_layout(aclDataLayout);
    output.info()->set_data_layout(aclDataLayout);

    {
        ARMNN_SCOPED_PROFILING_EVENT(Compute::Undefined, "ClChannelShuffleWorkload_configure");
        m_ChannelShuffleLayer.configure(clCompileContext, &input, &output, descriptor.m_Parameters.m_NumGroups);
    }
}

void ClChannelShuffleWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_CL_GUID("ClChannelShuffleWorkload_Execute", this->GetGuid());
    RunClFunction(m_ChannelShuffleLayer, CHECK_LOCATION());
}

} // namespace armnn
