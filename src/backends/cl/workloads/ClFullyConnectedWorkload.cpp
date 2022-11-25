//
// Copyright © 2017,2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ClFullyConnectedWorkload.hpp"
#include <cl/ClTensorHandle.hpp>
#include <armnn/backends/TensorHandle.hpp>
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
        // Same for bias as weights. We don't currently support non const.
        if (!biases.value().IsConstant())
        {
            return arm_compute::Status{arm_compute::ErrorCode::RUNTIME_ERROR,
                                        "Arm NN ClFullyConnectedWorkload does not support non constant bias."};
        }
        aclBiases = BuildArmComputeTensorInfo(biases.value());
        aclBiases.set_are_values_constant(biases.value().IsConstant());
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

ClFullyConnectedWorkload::ClFullyConnectedWorkload(
    const FullyConnectedQueueDescriptor& descriptor,
    const WorkloadInfo& info,
    std::shared_ptr<arm_compute::MemoryManagerOnDemand>& memoryManager,
    const arm_compute::CLCompileContext& clCompileContext)
    : ClBaseWorkload<FullyConnectedQueueDescriptor>(descriptor, info), m_FullyConnectedLayer(memoryManager)
{
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
    ARMNN_REPORT_PROFILING_WORKLOAD_DESC("ClFullyConnectedWorkload_Construct",
                                         descriptor.m_Parameters,
                                         detailsInfo,
                                         this->GetGuid());

    m_Data.ValidateInputsOutputs("ClFullyConnectedWorkload", descriptor.m_Parameters.GetNumInputs(),
                                 1);

    arm_compute::ICLTensor& input = static_cast<IClTensorHandle*>(m_Data.m_Inputs[0])->GetTensor();
    arm_compute::ICLTensor& output = static_cast<IClTensorHandle*>(m_Data.m_Outputs[0])->GetTensor();
    arm_compute::ICLTensor& weights = PolymorphicDowncast<IClTensorHandle*>(m_Data.m_Inputs[1])->GetTensor();

    arm_compute::ICLTensor* bias  = nullptr;
    if (m_Data.m_Parameters.m_BiasEnabled)
    {
        bias = &PolymorphicDowncast<IClTensorHandle*>(m_Data.m_Inputs[2])->GetTensor();
    }

    const arm_compute::ActivationLayerInfo activationInfo = ConvertAdditionalInfoToAclActivationLayerInfo(descriptor);

    arm_compute::FullyConnectedLayerInfo fc_info =
            ConvertFullyConnectedDescriptorToAclFullyConnectedLayerInfo(descriptor.m_Parameters,
                                                                        activationInfo);

    {
        ARMNN_SCOPED_PROFILING_EVENT(Compute::Undefined, "ClFullyConnectedWorkload_configure");
        m_FullyConnectedLayer.configure(clCompileContext,
                                        &input,
                                        &weights,
                                        bias,
                                        &output,
                                        fc_info);
    }
}

void ClFullyConnectedWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_CL_GUID("ClFullyConnectedWorkload_Execute", this->GetGuid());
    RunClFunction(m_FullyConnectedLayer, CHECK_LOCATION());
}

} //namespace armnn
