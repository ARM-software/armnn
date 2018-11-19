//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "NeonPooling2dWorkload.hpp"
#include <neon/NeonLayerSupport.hpp>
#include <neon/NeonTensorHandle.hpp>
#include <aclCommon/ArmComputeUtils.hpp>
#include <aclCommon/ArmComputeTensorUtils.hpp>

namespace armnn
{
using namespace armcomputetensorutils;

arm_compute::Status NeonPooling2dWorkloadValidate(const TensorInfo& input,
    const TensorInfo& output,
    const Pooling2dDescriptor& descriptor)
{
    const arm_compute::TensorInfo aclInputInfo =
            BuildArmComputeTensorInfo(input, descriptor.m_DataLayout);
    const arm_compute::TensorInfo aclOutputInfo =
            BuildArmComputeTensorInfo(output, descriptor.m_DataLayout);

    arm_compute::PoolingLayerInfo layerInfo = BuildArmComputePoolingLayerInfo(descriptor);

    return arm_compute::NEPoolingLayer::validate(&aclInputInfo, &aclOutputInfo, layerInfo);
}

NeonPooling2dWorkload::NeonPooling2dWorkload(
    const Pooling2dQueueDescriptor& descriptor, const WorkloadInfo& info)
    : BaseWorkload<Pooling2dQueueDescriptor>(descriptor, info)
{
    m_Data.ValidateInputsOutputs("NeonPooling2dWorkload", 1, 1);

    arm_compute::ITensor& input = boost::polymorphic_downcast<INeonTensorHandle*>(m_Data.m_Inputs[0])->GetTensor();
    arm_compute::ITensor& output = boost::polymorphic_downcast<INeonTensorHandle*>(m_Data.m_Outputs[0])->GetTensor();

    arm_compute::DataLayout aclDataLayout = ConvertDataLayout(m_Data.m_Parameters.m_DataLayout);
    input.info()->set_data_layout(aclDataLayout);
    output.info()->set_data_layout(aclDataLayout);

    arm_compute::PoolingLayerInfo layerInfo = BuildArmComputePoolingLayerInfo(m_Data.m_Parameters);

    m_PoolingLayer.configure(&input, &output, layerInfo);
}

void NeonPooling2dWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_NEON("NeonPooling2dWorkload_Execute");
    m_PoolingLayer.run();
}

} //namespace armnn
