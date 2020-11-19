//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "NeonFullyConnectedWorkload.hpp"

#include "NeonWorkloadUtils.hpp"

#include <aclCommon/ArmComputeTensorUtils.hpp>
#include <aclCommon/ArmComputeUtils.hpp>

#include <armnn/utility/PolymorphicDowncast.hpp>

#include <backendsCommon/CpuTensorHandle.hpp>

#include <arm_compute/runtime/NEON/functions/NEFullyConnectedLayer.h>

namespace armnn
{
using namespace armcomputetensorutils;

arm_compute::Status NeonFullyConnectedWorkloadValidate(const TensorInfo& input,
                                                       const TensorInfo& output,
                                                       const TensorInfo& weights,
                                                       const TensorInfo& biases,
                                                       const FullyConnectedDescriptor& descriptor,
                                                       const ActivationDescriptor* activationDescriptor)
{
    if (activationDescriptor)
    {
        std::vector<ActivationFunction> activations = {ActivationFunction::ReLu, ActivationFunction::BoundedReLu};
        if (std::find(activations.begin(), activations.end(), activationDescriptor->m_Function) == activations.end())
        {
            return arm_compute::Status{
                arm_compute::ErrorCode::RUNTIME_ERROR, "NeonFullyConnectedWorkload :Unsupported Activation Function"};
        }
    }

    const arm_compute::TensorInfo aclInput = BuildArmComputeTensorInfo(input);
    const arm_compute::TensorInfo aclOutput = BuildArmComputeTensorInfo(output);
    const arm_compute::TensorInfo aclWeights = BuildArmComputeTensorInfo(weights);

    arm_compute::TensorInfo aclBiases;
    arm_compute::TensorInfo *optionalAclBiases = nullptr;
    if (descriptor.m_BiasEnabled)
    {
        aclBiases  = BuildArmComputeTensorInfo(biases);
        optionalAclBiases = &aclBiases;
    }

    const arm_compute::FullyConnectedLayerInfo fullyConnectedLayerInfo =
        ConvertFullyConnectedDescriptorToAclFullyConnectedLayerInfo(descriptor, activationDescriptor);

    return arm_compute::NEFullyConnectedLayer::validate(&aclInput,
                                                        &aclWeights,
                                                        optionalAclBiases,
                                                        &aclOutput,
                                                        fullyConnectedLayerInfo);
}

NeonFullyConnectedWorkload::NeonFullyConnectedWorkload(const FullyConnectedQueueDescriptor& descriptor,
    const WorkloadInfo& info, std::shared_ptr<arm_compute::MemoryManagerOnDemand>& memoryManager)
    : BaseWorkload<FullyConnectedQueueDescriptor>(descriptor, info)
{
    m_Data.ValidateInputsOutputs("NeonFullyConnectedWorkload", 1, 1);

    arm_compute::ITensor& input = PolymorphicDowncast<IAclTensorHandle*>(m_Data.m_Inputs[0])->GetTensor();
    arm_compute::ITensor& output = PolymorphicDowncast<IAclTensorHandle*>(m_Data.m_Outputs[0])->GetTensor();

    m_WeightsTensor = std::make_unique<arm_compute::Tensor>();
    BuildArmComputeTensor(*m_WeightsTensor, m_Data.m_Weight->GetTensorInfo());

    if (m_Data.m_Parameters.m_BiasEnabled)
    {
        m_BiasesTensor = std::make_unique<arm_compute::Tensor>();
        BuildArmComputeTensor(*m_BiasesTensor, m_Data.m_Bias->GetTensorInfo());
    }

    const arm_compute::ActivationLayerInfo activationInfo = ConvertAdditionalInfoToAclActivationLayerInfo(descriptor);

    arm_compute::FullyConnectedLayerInfo fc_info  =
            ConvertFullyConnectedDescriptorToAclFullyConnectedLayerInfo(descriptor.m_Parameters, activationInfo);

    auto layer = std::make_unique<arm_compute::NEFullyConnectedLayer>(memoryManager);
    layer->configure(&input, m_WeightsTensor.get(), m_BiasesTensor.get(), &output, fc_info);
    m_FullyConnectedLayer.reset(layer.release());

    // Allocate
    if (m_Data.m_Weight->GetTensorInfo().GetDataType() == DataType::QAsymmU8)
    {
        InitializeArmComputeTensorData(*m_WeightsTensor, m_Data.m_Weight);
    }
    else
    {
        InitializeArmComputeTensorData(*m_WeightsTensor, m_Data.m_Weight);
    }

    if (m_BiasesTensor)
    {
        if (m_Data.m_Bias->GetTensorInfo().GetDataType() == DataType::Signed32)
        {
            InitializeArmComputeTensorData(*m_BiasesTensor, m_Data.m_Bias);
        }
        else
        {
            InitializeArmComputeTensorData(*m_BiasesTensor, m_Data.m_Bias);
        }
    }

    // Force Compute Library to perform the necessary copying and reshaping, after which
    // delete all the input tensors that will no longer be needed
    m_FullyConnectedLayer->prepare();
    FreeUnusedTensors();
}

void NeonFullyConnectedWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_NEON("NeonFullyConnectedWorkload_Execute");
    m_FullyConnectedLayer->run();
}

void NeonFullyConnectedWorkload::FreeUnusedTensors()
{
    FreeTensorIfUnused(m_WeightsTensor);
    FreeTensorIfUnused(m_BiasesTensor);
}

} //namespace armnn
