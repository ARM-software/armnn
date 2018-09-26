//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ClPooling2dBaseWorkload.hpp"
#include <backends/cl/ClLayerSupport.hpp>
#include <backends/cl/ClTensorHandle.hpp>
#include <backends/aclCommon/ArmComputeUtils.hpp>
#include <backends/aclCommon/ArmComputeTensorUtils.hpp>

namespace armnn
{
using namespace armcomputetensorutils;

arm_compute::Status ClPooling2dWorkloadValidate(const TensorInfo& input,
    const TensorInfo& output,
    const Pooling2dDescriptor& descriptor)
{
    const arm_compute::TensorInfo aclInputInfo = BuildArmComputeTensorInfo(input);
    const arm_compute::TensorInfo aclOutputInfo = BuildArmComputeTensorInfo(output);

    arm_compute::PoolingLayerInfo layerInfo = BuildArmComputePoolingLayerInfo(descriptor);

    return arm_compute::CLPoolingLayer::validate(&aclInputInfo, &aclOutputInfo, layerInfo);
}

template <armnn::DataType... dataTypes>
ClPooling2dBaseWorkload<dataTypes...>::ClPooling2dBaseWorkload(
    const Pooling2dQueueDescriptor& descriptor, const WorkloadInfo& info, const std::string& name)
    : TypedWorkload<Pooling2dQueueDescriptor, dataTypes...>(descriptor, info)
{
    m_Data.ValidateInputsOutputs(name, 1, 1);

    arm_compute::ICLTensor& input = static_cast<IClTensorHandle*>(m_Data.m_Inputs[0])->GetTensor();
    arm_compute::ICLTensor& output = static_cast<IClTensorHandle*>(m_Data.m_Outputs[0])->GetTensor();

    arm_compute::PoolingLayerInfo layerInfo = BuildArmComputePoolingLayerInfo(m_Data.m_Parameters);

    // Run the layer.
    m_PoolingLayer.configure(&input, &output, layerInfo);
}

template class ClPooling2dBaseWorkload<DataType::Float16, DataType::Float32>;
template class ClPooling2dBaseWorkload<DataType::QuantisedAsymm8>;

}
