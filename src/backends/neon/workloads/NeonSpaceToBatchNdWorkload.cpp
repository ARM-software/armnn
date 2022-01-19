//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "NeonSpaceToBatchNdWorkload.hpp"

#include "NeonWorkloadUtils.hpp"

#include <armnn/utility/NumericCast.hpp>
#include <armnn/utility/PolymorphicDowncast.hpp>

#include <ResolveType.hpp>

namespace armnn
{

using namespace armcomputetensorutils;

arm_compute::Status NeonSpaceToBatchNdWorkloadValidate(const TensorInfo& input,
                                                       const TensorInfo& output,
                                                       const SpaceToBatchNdDescriptor& descriptor)
{
    const arm_compute::TensorInfo aclInputInfo  = BuildArmComputeTensorInfo(input, descriptor.m_DataLayout);
    const arm_compute::TensorInfo aclOutputInfo = BuildArmComputeTensorInfo(output, descriptor.m_DataLayout);

    // ArmNN blockShape is [H, W] Cl asks for W, H
    int32_t blockHeight = armnn::numeric_cast<int32_t>(descriptor.m_BlockShape[0]);
    int32_t blockWidth  = armnn::numeric_cast<int32_t>(descriptor.m_BlockShape[1]);

    arm_compute::Size2D paddingLeftTop = BuildArmComputeSize2D(
            descriptor.m_PadList[1].first, descriptor.m_PadList[0].first);
    arm_compute::Size2D paddingRightBottom  = BuildArmComputeSize2D(
            descriptor.m_PadList[1].second, descriptor.m_PadList[0].second);

    return arm_compute::NESpaceToBatchLayer::validate(&aclInputInfo,
                                                      blockWidth,
                                                      blockHeight,
                                                      paddingLeftTop,
                                                      paddingRightBottom,
                                                      &aclOutputInfo);
}

NeonSpaceToBatchNdWorkload::NeonSpaceToBatchNdWorkload(const SpaceToBatchNdQueueDescriptor& descriptor,
                                                       const WorkloadInfo& info)
        : NeonBaseWorkload<SpaceToBatchNdQueueDescriptor>(descriptor, info)
{
    // Report Profiling Details
    ARMNN_REPORT_PROFILING_WORKLOAD_DESC("NeonSpaceToBatchNdWorkload_Construct",
                                         descriptor.m_Parameters,
                                         info,
                                         this->GetGuid());

    m_Data.ValidateInputsOutputs("NESpaceToBatchNdWorkload", 1, 1);

    arm_compute::ITensor& input  =
            PolymorphicPointerDowncast<IAclTensorHandle>(m_Data.m_Inputs[0])->GetTensor();
    arm_compute::ITensor& output =
            PolymorphicPointerDowncast<IAclTensorHandle>(m_Data.m_Outputs[0])->GetTensor();

    // ArmNN blockShape is [H, W] Cl asks for W, H
    int32_t blockHeight = armnn::numeric_cast<int32_t>(m_Data.m_Parameters.m_BlockShape[0]);
    int32_t blockWidth  = armnn::numeric_cast<int32_t>(m_Data.m_Parameters.m_BlockShape[1]);

    arm_compute::Size2D paddingLeftTop = BuildArmComputeSize2D(
            m_Data.m_Parameters.m_PadList[1].first, m_Data.m_Parameters.m_PadList[0].first);
    arm_compute::Size2D paddingRightBottom  = BuildArmComputeSize2D(
            m_Data.m_Parameters.m_PadList[1].second, m_Data.m_Parameters.m_PadList[0].second);

    arm_compute::DataLayout aclDataLayout = ConvertDataLayout(m_Data.m_Parameters.m_DataLayout);
    input.info()->set_data_layout(aclDataLayout);
    output.info()->set_data_layout(aclDataLayout);

    m_Layer.reset(new arm_compute::NESpaceToBatchLayer());
    m_Layer->configure(&input,
                       blockWidth,
                       blockHeight,
                       paddingLeftTop,
                       paddingRightBottom,
                       &output);
    m_Layer->prepare();
}

void NeonSpaceToBatchNdWorkload::Execute() const
{
    if (m_Layer)
    {
        ARMNN_SCOPED_PROFILING_EVENT_NEON_GUID("NeonSpaceToBatchNdWorkload_Execute", this->GetGuid());
        m_Layer->run();
    }
}

} //namespace armnn