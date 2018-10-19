//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ClPooling2dWorkload.hpp"
#include <backends/cl/ClLayerSupport.hpp>
#include <backends/cl/ClTensorHandle.hpp>
#include <backends/aclCommon/ArmComputeUtils.hpp>
#include <backends/aclCommon/ArmComputeTensorUtils.hpp>

#include "ClWorkloadUtils.hpp"

namespace armnn
{
using namespace armcomputetensorutils;

arm_compute::Status ClPooling2dWorkloadValidate(const TensorInfo& input,
    const TensorInfo& output,
    const Pooling2dDescriptor& descriptor)
{
    const arm_compute::TensorInfo aclInputInfo =
            BuildArmComputeTensorInfo(input, descriptor.m_DataLayout.GetDataLayout());
    const arm_compute::TensorInfo aclOutputInfo =
            BuildArmComputeTensorInfo(output, descriptor.m_DataLayout.GetDataLayout());

    arm_compute::PoolingLayerInfo layerInfo = BuildArmComputePoolingLayerInfo(descriptor);

    return arm_compute::CLPoolingLayer::validate(&aclInputInfo, &aclOutputInfo, layerInfo);
}

ClPooling2dWorkload::ClPooling2dWorkload(
    const Pooling2dQueueDescriptor& descriptor, const WorkloadInfo& info)
    : BaseWorkload<Pooling2dQueueDescriptor>(descriptor, info)
{
    m_Data.ValidateInputsOutputs("ClPooling2dWorkload", 1, 1);

    arm_compute::ICLTensor& input = static_cast<IClTensorHandle*>(m_Data.m_Inputs[0])->GetTensor();
    arm_compute::ICLTensor& output = static_cast<IClTensorHandle*>(m_Data.m_Outputs[0])->GetTensor();

    arm_compute::DataLayout aclDataLayout = ConvertDataLayout(m_Data.m_Parameters.m_DataLayout.GetDataLayout());
    input.info()->set_data_layout(aclDataLayout);
    output.info()->set_data_layout(aclDataLayout);

    arm_compute::PoolingLayerInfo layerInfo = BuildArmComputePoolingLayerInfo(m_Data.m_Parameters);

    // Run the layer.
    m_PoolingLayer.configure(&input, &output, layerInfo);
}

void ClPooling2dWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_CL("ClPooling2dWorkload_Execute");
    RunClFunction(m_PoolingLayer, CHECK_LOCATION());
}

}
