//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "NeonPooling2dBaseWorkload.hpp"
#include <backends/neon/NeonLayerSupport.hpp>
#include <backends/neon/NeonTensorHandle.hpp>
#include <backends/aclCommon/ArmComputeUtils.hpp>
#include <backends/aclCommon/ArmComputeTensorUtils.hpp>

namespace armnn
{
using namespace armcomputetensorutils;

arm_compute::Status NeonPooling2dWorkloadValidate(const TensorInfo& input,
    const TensorInfo& output,
    const Pooling2dDescriptor& descriptor)
{
    const arm_compute::TensorInfo aclInputInfo = BuildArmComputeTensorInfo(input);
    const arm_compute::TensorInfo aclOutputInfo = BuildArmComputeTensorInfo(output);

    arm_compute::PoolingLayerInfo layerInfo = BuildArmComputePoolingLayerInfo(descriptor);

    return arm_compute::NEPoolingLayer::validate(&aclInputInfo, &aclOutputInfo, layerInfo);
}

template <armnn::DataType... dataTypes>
NeonPooling2dBaseWorkload<dataTypes...>::NeonPooling2dBaseWorkload(
    const Pooling2dQueueDescriptor& descriptor, const WorkloadInfo& info, const std::string& name)
    : TypedWorkload<Pooling2dQueueDescriptor, dataTypes...>(descriptor, info)
{
    m_Data.ValidateInputsOutputs(name, 1, 1);

    arm_compute::ITensor& input = boost::polymorphic_downcast<INeonTensorHandle*>(m_Data.m_Inputs[0])->GetTensor();
    arm_compute::ITensor& output = boost::polymorphic_downcast<INeonTensorHandle*>(m_Data.m_Outputs[0])->GetTensor();

    arm_compute::PoolingLayerInfo layerInfo = BuildArmComputePoolingLayerInfo(m_Data.m_Parameters);

    m_PoolingLayer.configure(&input, &output, layerInfo);
}

template class NeonPooling2dBaseWorkload<DataType::Float16, DataType::Float32>;
template class NeonPooling2dBaseWorkload<DataType::QuantisedAsymm8>;

} //namespace armnn

