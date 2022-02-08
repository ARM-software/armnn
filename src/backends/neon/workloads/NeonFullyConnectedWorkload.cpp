//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
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
    const arm_compute::TensorInfo aclWeights = BuildArmComputeTensorInfo(weights);

    arm_compute::TensorInfo aclBiases;
    arm_compute::TensorInfo* optionalAclBiases = nullptr;
    if (descriptor.m_BiasEnabled)
    {
        ARMNN_ASSERT(biases.has_value());
        aclBiases = BuildArmComputeTensorInfo(biases.value());
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

    arm_compute::FullyConnectedLayerInfo fc_info =
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

    // Add details for profiling output
    WorkloadInfo detailsInfo;

    detailsInfo.m_InputTensorInfos = info.m_InputTensorInfos;
    detailsInfo.m_OutputTensorInfos = info.m_OutputTensorInfos;
    detailsInfo.m_WeightsTensorInfo = armnn::Optional<armnn::TensorInfo>(descriptor.m_Weight->GetTensorInfo());
    if (descriptor.m_Parameters.m_BiasEnabled)
    {
        detailsInfo.m_BiasTensorInfo = armnn::Optional<armnn::TensorInfo>(descriptor.m_Bias->GetTensorInfo());
    }

    // Report Profiling Details
    ARMNN_REPORT_PROFILING_WORKLOAD_DESC("NeonFullyConnectedWorkload_Construct",
                                         descriptor.m_Parameters,
                                         detailsInfo,
                                         this->GetGuid());

    // Force Compute Library to perform the necessary copying and reshaping, after which
    // delete all the input tensors that will no longer be needed
    m_FullyConnectedLayer->prepare();
    FreeUnusedTensors();
}

void NeonFullyConnectedWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_NEON_GUID("NeonFullyConnectedWorkload_Execute", this->GetGuid());
    m_FullyConnectedLayer->run();
}

void NeonFullyConnectedWorkload::FreeUnusedTensors()
{
    FreeTensorIfUnused(m_WeightsTensor);
    FreeTensorIfUnused(m_BiasesTensor);
}

} //namespace armnn
