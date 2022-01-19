//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ClSpaceToBatchNdWorkload.hpp"

#include "ClWorkloadUtils.hpp"

#include <aclCommon/ArmComputeUtils.hpp>
#include <aclCommon/ArmComputeTensorUtils.hpp>
#include <armnn/utility/NumericCast.hpp>
#include <armnn/utility/PolymorphicDowncast.hpp>
#include <armnn/backends/TensorHandle.hpp>
#include <cl/ClLayerSupport.hpp>
#include <cl/ClTensorHandle.hpp>
#include <cl/ClLayerSupport.hpp>

namespace armnn
{
using namespace armcomputetensorutils;

arm_compute::Status ClSpaceToBatchNdWorkloadValidate(const TensorInfo& input,
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

    return arm_compute::CLSpaceToBatchLayer::validate(&aclInputInfo,
                                                      blockWidth,
                                                      blockHeight,
                                                      paddingLeftTop,
                                                      paddingRightBottom,
                                                      &aclOutputInfo);
}

ClSpaceToBatchNdWorkload::ClSpaceToBatchNdWorkload(
    const SpaceToBatchNdQueueDescriptor& descriptor,
    const WorkloadInfo& info,
    const arm_compute::CLCompileContext& clCompileContext)
    : ClBaseWorkload<SpaceToBatchNdQueueDescriptor>(descriptor, info)
{
    // Report Profiling Details
    ARMNN_REPORT_PROFILING_WORKLOAD_DESC("ClSpaceToBatchNdWorkload_Construct",
                                         descriptor.m_Parameters,
                                         info,
                                         this->GetGuid());

    m_Data.ValidateInputsOutputs("ClSpaceToBatchNdWorkload", 1, 1);

    arm_compute::ICLTensor& input  =
        armnn::PolymorphicPointerDowncast<IClTensorHandle>(m_Data.m_Inputs[0])->GetTensor();
    arm_compute::ICLTensor& output =
        armnn::PolymorphicPointerDowncast<IClTensorHandle>(m_Data.m_Outputs[0])->GetTensor();

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

    {
        ARMNN_SCOPED_PROFILING_EVENT(Compute::Undefined, "ClSpaceToBatchNdWorkload_configure");
        m_SpaceToBatchLayer.configure(clCompileContext,
                                      &input,
                                      blockWidth,
                                      blockHeight,
                                      paddingLeftTop,
                                      paddingRightBottom,
                                      &output);
    }
}

void ClSpaceToBatchNdWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_CL_GUID("ClSpaceToBatchNdWorkload_Execute", this->GetGuid());
    RunClFunction(m_SpaceToBatchLayer, CHECK_LOCATION());
}

} //namespace armnn
