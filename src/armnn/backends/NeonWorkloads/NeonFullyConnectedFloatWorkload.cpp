//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//

#include "NeonFullyConnectedFloatWorkload.hpp"

#include "backends/ArmComputeTensorUtils.hpp"
#include "backends/ArmComputeUtils.hpp"
#include "backends/CpuTensorHandle.hpp"

namespace armnn
{
using namespace armcomputetensorutils;

arm_compute::Status NeonFullyConnectedWorkloadValidate(const TensorInfo& input,
                                                       const TensorInfo& output,
                                                       const TensorInfo& weights,
                                                       const TensorInfo& biases,
                                                       const FullyConnectedDescriptor& descriptor)
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
        ConvertFullyConnectedDescriptorToAclFullyConnectedLayerInfo(descriptor);


    return arm_compute::NEFullyConnectedLayer::validate(&aclInput,
                                                        &aclWeights,
                                                        optionalAclBiases,
                                                        &aclOutput,
                                                        fullyConnectedLayerInfo);
}

NeonFullyConnectedFloatWorkload::NeonFullyConnectedFloatWorkload(const FullyConnectedQueueDescriptor& descriptor,
    const WorkloadInfo& info, std::shared_ptr<arm_compute::MemoryManagerOnDemand>& memoryManager)
    : FloatWorkload<FullyConnectedQueueDescriptor>(descriptor, info)
    , m_FullyConnectedLayer(memoryManager)
{
    m_Data.ValidateInputsOutputs("NeonFullyConnectedFloatWorkload", 1, 1);

    arm_compute::ITensor& input = boost::polymorphic_downcast<INeonTensorHandle*>(m_Data.m_Inputs[0])->GetTensor();
    arm_compute::ITensor& output = boost::polymorphic_downcast<INeonTensorHandle*>(m_Data.m_Outputs[0])->GetTensor();

    m_WeightsTensor = std::make_unique<arm_compute::Tensor>();
    BuildArmComputeTensor(*m_WeightsTensor, m_Data.m_Weight->GetTensorInfo());

    if (m_Data.m_Parameters.m_BiasEnabled)
    {
        m_BiasesTensor = std::make_unique<arm_compute::Tensor>();
        BuildArmComputeTensor(*m_BiasesTensor, m_Data.m_Bias->GetTensorInfo());
    }

    // Construct
    arm_compute::FullyConnectedLayerInfo fc_info;
    fc_info.transpose_weights = m_Data.m_Parameters.m_TransposeWeightMatrix;
    m_FullyConnectedLayer.configure(&input, m_WeightsTensor.get(), m_BiasesTensor.get(), &output, fc_info);

    // Allocate
    InitializeArmComputeTensorDataForFloatTypes(*m_WeightsTensor, m_Data.m_Weight);

    if (m_BiasesTensor)
    {
        InitializeArmComputeTensorDataForFloatTypes(*m_BiasesTensor, m_Data.m_Bias);
    }

    // Force Compute Library to perform the necessary copying and reshaping, after which
    // delete all the input tensors that will no longer be needed
    m_FullyConnectedLayer.prepare();
    FreeUnusedTensors();
}

void NeonFullyConnectedFloatWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_NEON("NeonFullyConnectedFloatWorkload_Execute");
    m_FullyConnectedLayer.run();
}

void NeonFullyConnectedFloatWorkload::FreeUnusedTensors()
{
    FreeTensorIfUnused(m_WeightsTensor);
    FreeTensorIfUnused(m_BiasesTensor);
}

} //namespace armnn

