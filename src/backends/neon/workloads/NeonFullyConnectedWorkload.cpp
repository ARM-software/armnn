//
// Copyright © 2017,2022-2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "NeonFullyConnectedWorkload.hpp"

#include "NeonWorkloadUtils.hpp"

#include <aclCommon/ArmComputeTensorUtils.hpp>
#include <aclCommon/ArmComputeUtils.hpp>

#include <armnn/utility/PolymorphicDowncast.hpp>

#include <armnn/backends/TensorHandle.hpp>

#include <arm_compute/runtime/NEON/functions/NEFullyConnectedLayer.h>

namespace armnn
{
using namespace armcomputetensorutils;
using ACLMemManagerOnDemand = std::shared_ptr<arm_compute::MemoryManagerOnDemand>;

arm_compute::Status NeonFullyConnectedWorkloadValidate(const TensorInfo& input,
                                                       const TensorInfo& output,
                                                       const TensorInfo& weights,
                                                       const Optional<TensorInfo>& biases,
                                                       const FullyConnectedDescriptor& descriptor,
                                                       const ActivationDescriptor* activationDescriptor)
{
    const arm_compute::TensorInfo aclInput = BuildArmComputeTensorInfo(input);
    const arm_compute::TensorInfo aclOutput = BuildArmComputeTensorInfo(output);
    arm_compute::TensorInfo aclWeights = BuildArmComputeTensorInfo(weights);
    aclWeights.set_are_values_constant(weights.IsConstant());

    arm_compute::TensorInfo aclBiases;
    arm_compute::TensorInfo* optionalAclBiases = nullptr;
    if (descriptor.m_BiasEnabled)
    {
        ARMNN_ASSERT(biases.has_value());
        aclBiases = BuildArmComputeTensorInfo(biases.value());
        aclBiases.set_are_values_constant(biases.value().IsConstant());
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
                                                       const WorkloadInfo& info,
                                                       ACLMemManagerOnDemand& memoryManager)
    : NeonBaseWorkload<FullyConnectedQueueDescriptor>(descriptor, info)
{
    m_Data.ValidateInputsOutputs("NeonFullyConnectedWorkload", descriptor.m_Parameters.GetNumInputs(), 1);

    arm_compute::ITensor& input = PolymorphicDowncast<IAclTensorHandle*>(m_Data.m_Inputs[0])->GetTensor();
    arm_compute::ITensor& weights = PolymorphicDowncast<IAclTensorHandle*>(m_Data.m_Inputs[1])->GetTensor();
    m_WeightsTensorInfo = info.m_InputTensorInfos[1];
    weights.info()->set_are_values_constant(m_WeightsTensorInfo.IsConstant());
    arm_compute::ITensor& output = PolymorphicDowncast<IAclTensorHandle*>(m_Data.m_Outputs[0])->GetTensor();
    if (m_WeightsTensorInfo.IsConstant())
    {
        // Copy the weights' tensor into arm_compute tensor.
        m_WeightsTensor = std::make_unique<arm_compute::Tensor>();
        BuildArmComputeTensor(*m_WeightsTensor, m_WeightsTensorInfo);
        m_WeightsTensor->info()->set_are_values_constant(m_WeightsTensorInfo.IsConstant());
    }

    if (m_Data.m_Parameters.m_BiasEnabled)
    {
        // Copy the biases tensor into arm_compute tensor.
        m_BiasesTensor = std::make_unique<arm_compute::Tensor>();
        m_BiasesTensorInfo = info.m_InputTensorInfos[2];
        BuildArmComputeTensor(*m_BiasesTensor, m_BiasesTensorInfo);
        m_BiasesTensor->info()->set_are_values_constant(m_BiasesTensorInfo.IsConstant());
    }

    const arm_compute::ActivationLayerInfo activationInfo = ConvertAdditionalInfoToAclActivationLayerInfo(descriptor);
    arm_compute::FullyConnectedLayerInfo fc_info =
        ConvertFullyConnectedDescriptorToAclFullyConnectedLayerInfo(descriptor.m_Parameters, activationInfo);

    auto layer = std::make_unique<arm_compute::NEFullyConnectedLayer>(memoryManager);
    if (m_WeightsTensorInfo.IsConstant())
    {
        layer->configure(&input, m_WeightsTensor.get(), m_BiasesTensor.get(), &output, fc_info);
    }
    else
    {
        layer->configure(&input, &weights, m_BiasesTensor.get(), &output, fc_info);
    }
    m_FullyConnectedLayer.reset(layer.release());

    // Add details for profiling output
    WorkloadInfo detailsInfo;

    detailsInfo.m_InputTensorInfos = info.m_InputTensorInfos;
    detailsInfo.m_OutputTensorInfos = info.m_OutputTensorInfos;
    detailsInfo.m_WeightsTensorInfo = armnn::Optional<armnn::TensorInfo>(info.m_InputTensorInfos[1]);
    if (descriptor.m_Parameters.m_BiasEnabled)
    {
        detailsInfo.m_BiasTensorInfo = armnn::Optional<armnn::TensorInfo>(info.m_InputTensorInfos[2]);
    }

    // Report Profiling Details
    ARMNN_REPORT_PROFILING_WORKLOAD_DESC("NeonFullyConnectedWorkload_Construct",
                                         descriptor.m_Parameters,
                                         detailsInfo,
                                         this->GetGuid());
}

void NeonFullyConnectedWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_NEON_GUID("NeonFullyConnectedWorkload_Execute", this->GetGuid());
    // The constant tensors may not be fully in place until the workload is Executed
    if (!prepared)
    {
        if (m_WeightsTensorInfo.IsConstant())
        {
            InitializeArmComputeTensorData(*m_WeightsTensor, m_WeightsTensorInfo, m_Data.m_Inputs[1]);
            m_WeightsTensor->info()->set_are_values_constant(m_WeightsTensorInfo.IsConstant());
        }

        if (m_Data.m_Parameters.m_BiasEnabled)
        {
            InitializeArmComputeTensorData(*m_BiasesTensor, m_BiasesTensorInfo, m_Data.m_Inputs[2]);
            m_BiasesTensor->info()->set_are_values_constant(m_BiasesTensorInfo.IsConstant());
        }
        if (m_WeightsTensorInfo.IsConstant())
        {
            FreeTensorIfUnused(m_WeightsTensor);
        }
        if (m_BiasesTensorInfo.IsConstant())
        {
            FreeTensorIfUnused(m_BiasesTensor);
        }
        prepared = true;
    }
    m_FullyConnectedLayer->run();
}

} //namespace armnn
