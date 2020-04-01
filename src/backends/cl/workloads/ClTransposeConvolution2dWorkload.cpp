//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ClTransposeConvolution2dWorkload.hpp"

#include "ClWorkloadUtils.hpp"

#include <cl/ClLayerSupport.hpp>
#include <cl/ClTensorHandle.hpp>
#include <cl/ClLayerSupport.hpp>

#include <aclCommon/ArmComputeUtils.hpp>
#include <aclCommon/ArmComputeTensorUtils.hpp>

#include <backendsCommon/CpuTensorHandle.hpp>

#include <arm_compute/runtime/CL/functions/CLDeconvolutionLayer.h>

namespace armnn
{

using namespace armcomputetensorutils;

arm_compute::Status ClTransposeConvolution2dWorkloadValidate(const TensorInfo& input,
                                                             const TensorInfo& output,
                                                             const TransposeConvolution2dDescriptor& descriptor,
                                                             const TensorInfo& weights,
                                                             const Optional<TensorInfo>& biases)
{
    arm_compute::TensorInfo aclInputInfo   = BuildArmComputeTensorInfo(input, descriptor.m_DataLayout);
    arm_compute::TensorInfo aclOutputInfo  = BuildArmComputeTensorInfo(output, descriptor.m_DataLayout);
    arm_compute::TensorInfo aclWeightsInfo = BuildArmComputeTensorInfo(weights, descriptor.m_DataLayout);

    arm_compute::TensorInfo aclBiasesInfo;
    arm_compute::TensorInfo *optionalAclBiasesInfo = nullptr;

    if (descriptor.m_BiasEnabled)
    {
        ARMNN_ASSERT(biases.has_value());

        aclBiasesInfo = BuildArmComputeTensorInfo(biases.value(), descriptor.m_DataLayout);
        optionalAclBiasesInfo = &aclBiasesInfo;
    }

    arm_compute::PadStrideInfo padStrideInfo = BuildArmComputePadStrideInfo(descriptor);

    return arm_compute::CLDeconvolutionLayer::validate(&aclInputInfo,
                                                       &aclWeightsInfo,
                                                       optionalAclBiasesInfo,
                                                       &aclOutputInfo,
                                                       padStrideInfo);
}

ClTransposeConvolution2dWorkload::ClTransposeConvolution2dWorkload(
    const TransposeConvolution2dQueueDescriptor& descriptor,
    const WorkloadInfo& info,
    std::shared_ptr<arm_compute::MemoryManagerOnDemand>& memoryManager) :
    BaseWorkload<TransposeConvolution2dQueueDescriptor>(descriptor, info),
    m_Layer(memoryManager)
{
    const TensorInfo& weightInfo = m_Data.m_Weight->GetTensorInfo();

    m_WeightsTensor = std::make_unique<arm_compute::CLTensor>();
    BuildArmComputeTensor(*m_WeightsTensor, weightInfo, m_Data.m_Parameters.m_DataLayout);

    if (m_Data.m_Parameters.m_BiasEnabled)
    {
        m_BiasesTensor = std::make_unique<arm_compute::CLTensor>();
        BuildArmComputeTensor(*m_BiasesTensor, m_Data.m_Bias->GetTensorInfo(), m_Data.m_Parameters.m_DataLayout);
    }

    m_Data.ValidateInputsOutputs("ClTransposeConvolution2dWorkload", 1, 1);

    arm_compute::ICLTensor& input  = static_cast<IClTensorHandle*>(m_Data.m_Inputs[0])->GetTensor();
    arm_compute::ICLTensor& output = static_cast<IClTensorHandle*>(m_Data.m_Outputs[0])->GetTensor();

    arm_compute::DataLayout aclDataLayout = ConvertDataLayout(m_Data.m_Parameters.m_DataLayout);

    input.info()->set_data_layout(aclDataLayout);
    output.info()->set_data_layout(aclDataLayout);

    arm_compute::PadStrideInfo padStrideInfo = BuildArmComputePadStrideInfo(m_Data.m_Parameters);
    m_Layer.configure(&input, m_WeightsTensor.get(), m_BiasesTensor.get(), &output, padStrideInfo);

    InitializeArmComputeClTensorData(*m_WeightsTensor, m_Data.m_Weight);
    if (m_BiasesTensor)
    {
        InitializeArmComputeClTensorData(*m_BiasesTensor, m_Data.m_Bias);
    }

    m_Layer.prepare();

    FreeUnusedTensors();
}

void ClTransposeConvolution2dWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_CL("ClTransposeConvolution2dWorkload_Execute");
    RunClFunction(m_Layer, CHECK_LOCATION());
}

void ClTransposeConvolution2dWorkload::FreeUnusedTensors()
{
    FreeTensorIfUnused(m_WeightsTensor);
    FreeTensorIfUnused(m_BiasesTensor);
}

} // namespace armnn
