//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ClFullyConnectedWorkload.hpp"
#include <cl/ClTensorHandle.hpp>
#include <backendsCommon/CpuTensorHandle.hpp>
#include <aclCommon/ArmComputeTensorUtils.hpp>
#include <aclCommon/ArmComputeUtils.hpp>
#include <cl/ClLayerSupport.hpp>

#include "ClWorkloadUtils.hpp"

namespace armnn
{
using namespace armcomputetensorutils;

arm_compute::Status ClFullyConnectedWorkloadValidate(const TensorInfo& input,
                                                     const TensorInfo& output,
                                                     const TensorInfo& weights,
                                                     const TensorInfo& biases,
                                                     const FullyConnectedDescriptor& descriptor,
                                                     const ActivationDescriptor* activationDescriptor)
{
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

    return arm_compute::CLFullyConnectedLayer::validate(&aclInput,
                                                        &aclWeights,
                                                        optionalAclBiases,
                                                        &aclOutput,
                                                        fullyConnectedLayerInfo);
}

ClFullyConnectedWorkload::ClFullyConnectedWorkload(const FullyConnectedQueueDescriptor& descriptor,
    const WorkloadInfo& info, std::shared_ptr<arm_compute::MemoryManagerOnDemand>& memoryManager)
    : BaseWorkload<FullyConnectedQueueDescriptor>(descriptor, info)
    , m_FullyConnectedLayer(memoryManager)
{
    m_WeightsTensor = std::make_unique<arm_compute::CLTensor>();
    BuildArmComputeTensor(*m_WeightsTensor, m_Data.m_Weight->GetTensorInfo());

    if (m_Data.m_Parameters.m_BiasEnabled)
    {
        m_BiasesTensor = std::make_unique<arm_compute::CLTensor>();
        BuildArmComputeTensor(*m_BiasesTensor, m_Data.m_Bias->GetTensorInfo());
    }

    m_Data.ValidateInputsOutputs("ClFullyConnectedWorkload", 1, 1);

    arm_compute::ICLTensor& input  = static_cast<IClTensorHandle*>(m_Data.m_Inputs[0])->GetTensor();
    arm_compute::ICLTensor& output = static_cast<IClTensorHandle*>(m_Data.m_Outputs[0])->GetTensor();

    const arm_compute::ActivationLayerInfo activationInfo = ConvertAdditionalInfoToAclActivationLayerInfo(descriptor);

    arm_compute::FullyConnectedLayerInfo fc_info =
            ConvertFullyConnectedDescriptorToAclFullyConnectedLayerInfo(descriptor.m_Parameters, activationInfo);

    m_FullyConnectedLayer.configure(&input, m_WeightsTensor.get(), m_BiasesTensor.get(), &output, fc_info);

    InitializeArmComputeClTensorData(*m_WeightsTensor, m_Data.m_Weight);

    if (m_BiasesTensor)
    {
        InitializeArmComputeClTensorData(*m_BiasesTensor, m_Data.m_Bias);
    }

    // Force Compute Library to perform the necessary copying and reshaping, after which
    // delete all the input tensors that will no longer be needed
    m_FullyConnectedLayer.prepare();
    FreeUnusedTensors();
}

void ClFullyConnectedWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_CL("ClFullyConnectedWorkload_Execute");
    RunClFunction(m_FullyConnectedLayer, CHECK_LOCATION());
}

void ClFullyConnectedWorkload::FreeUnusedTensors()
{
    FreeTensorIfUnused(m_WeightsTensor);
    FreeTensorIfUnused(m_BiasesTensor);
}

} //namespace armnn
