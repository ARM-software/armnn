//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ClConvolution2dWorkload.hpp"

#include "ClWorkloadUtils.hpp"

#include <cl/ClLayerSupport.hpp>
#include <cl/ClTensorHandle.hpp>
#include <cl/ClLayerSupport.hpp>
#include <aclCommon/ArmComputeUtils.hpp>
#include <aclCommon/ArmComputeTensorUtils.hpp>
#include <armnn/backends/TensorHandle.hpp>

#include <arm_compute/runtime/CL/functions/CLConvolutionLayer.h>

namespace armnn
{
using namespace armcomputetensorutils;

arm_compute::Status ClConvolution2dWorkloadValidate(const TensorInfo& input,
                                                    const TensorInfo& output,
                                                    const Convolution2dDescriptor& descriptor,
                                                    const TensorInfo& weights,
                                                    const Optional<TensorInfo>& biases,
                                                    bool isFastMathEnabled,
                                                    const ActivationDescriptor* activationDescriptor)
{
    const arm_compute::TensorInfo aclInputInfo = BuildArmComputeTensorInfo(input, descriptor.m_DataLayout);
    const arm_compute::TensorInfo aclOutputInfo = BuildArmComputeTensorInfo(output, descriptor.m_DataLayout);
    const arm_compute::TensorInfo aclWeightsInfo = BuildArmComputeTensorInfo(weights, descriptor.m_DataLayout);

    const arm_compute::Size2D aclDilationInfo = BuildArmComputeSize2D(descriptor.m_DilationX,
                                                                      descriptor.m_DilationY);

    arm_compute::TensorInfo aclBiasesInfo;
    arm_compute::TensorInfo *optionalAclBiasesInfo = nullptr;

    if (descriptor.m_BiasEnabled)
    {
        ARMNN_ASSERT(biases.has_value());

        aclBiasesInfo = BuildArmComputeTensorInfo(biases.value(), descriptor.m_DataLayout);
        optionalAclBiasesInfo = &aclBiasesInfo;
    }

    arm_compute::PadStrideInfo layerInfo = BuildArmComputePadStrideInfo(descriptor);

    const arm_compute::ActivationLayerInfo activationInfo = ConvertActivationDescriptorToAclActivationLayerInfo(
            activationDescriptor);

    return arm_compute::CLConvolutionLayer::validate(&aclInputInfo,
                                                     &aclWeightsInfo,
                                                     optionalAclBiasesInfo,
                                                     &aclOutputInfo,
                                                     layerInfo,
                                                     arm_compute::WeightsInfo(),
                                                     aclDilationInfo,
                                                     activationInfo,
                                                     isFastMathEnabled);
}

ClConvolution2dWorkload::ClConvolution2dWorkload(const Convolution2dQueueDescriptor& descriptor,
                                                 const WorkloadInfo& info,
                                                 std::shared_ptr<arm_compute::MemoryManagerOnDemand>& memoryManager,
                                                 const arm_compute::CLCompileContext& clCompileContext,
                                                 const bool isFastMathEnabled)
    : ClBaseWorkload<Convolution2dQueueDescriptor>(descriptor, info)
    , m_ConvolutionLayer(memoryManager)
{
    ARMNN_SCOPED_PROFILING_EVENT(Compute::Undefined, "ClConvolution2dWorkload");
    const TensorInfo& weightInfo = m_Data.m_Weight->GetTensorInfo();

    m_KernelTensor = std::make_unique<arm_compute::CLTensor>();
    BuildArmComputeTensor(*m_KernelTensor, weightInfo, m_Data.m_Parameters.m_DataLayout);

    const arm_compute::Size2D aclDilationInfo = BuildArmComputeSize2D(m_Data.m_Parameters.m_DilationX,
                                                                      m_Data.m_Parameters.m_DilationY);

    if (m_Data.m_Parameters.m_BiasEnabled)
    {
        m_BiasTensor = std::make_unique<arm_compute::CLTensor>();
        BuildArmComputeTensor(*m_BiasTensor, m_Data.m_Bias->GetTensorInfo(), m_Data.m_Parameters.m_DataLayout);
    }

    m_Data.ValidateInputsOutputs("ClConvolution2dWorkload", 1, 1);

    arm_compute::ICLTensor& input  = static_cast<IClTensorHandle*>(m_Data.m_Inputs[0])->GetTensor();
    arm_compute::ICLTensor& output = static_cast<IClTensorHandle*>(m_Data.m_Outputs[0])->GetTensor();

    // Create Proxy tensor and set the initial tensor handle to it
    m_InputProxy = std::make_unique<ICLTensorProxy>(&input);
    m_OutputProxy = std::make_unique<ICLTensorProxy>(&output);

    arm_compute::DataLayout aclDataLayout = ConvertDataLayout(m_Data.m_Parameters.m_DataLayout);
    input.info()->set_data_layout(aclDataLayout);
    output.info()->set_data_layout(aclDataLayout);

    arm_compute::PadStrideInfo padStrideInfo = BuildArmComputePadStrideInfo(m_Data.m_Parameters);

    const arm_compute::ActivationLayerInfo activationInfo = ConvertAdditionalInfoToAclActivationLayerInfo(descriptor);

    {
        ARMNN_SCOPED_PROFILING_EVENT(Compute::Undefined, "ClConvolution2dWorkload_configure");
        m_ConvolutionLayer.configure(clCompileContext,
                                     m_InputProxy.get(),
                                     m_KernelTensor.get(),
                                     m_BiasTensor.get(),
                                     m_OutputProxy.get(),
                                     padStrideInfo,
                                     arm_compute::WeightsInfo(),
                                     aclDilationInfo,
                                     activationInfo,
                                     isFastMathEnabled);
    }

    m_ConvolutionMethod =
        m_ConvolutionLayer.get_convolution_method(input.info(),
                                                  m_KernelTensor->info(),
                                                  output.info(),
                                                  padStrideInfo,
                                                  arm_compute::WeightsInfo(),
                                                  activationInfo,
                                                  arm_compute::CLScheduler::get().target(),
                                                  aclDilationInfo,
                                                  isFastMathEnabled);

     // Add details for profiling output
    WorkloadInfo detailsInfo;

    detailsInfo.m_InputTensorInfos = info.m_InputTensorInfos;
    detailsInfo.m_OutputTensorInfos = info.m_OutputTensorInfos;
    detailsInfo.m_WeightsTensorInfo = armnn::Optional<armnn::TensorInfo>(descriptor.m_Weight->GetTensorInfo());
    detailsInfo.m_ConvolutionMethod = armnn::Optional<std::string>(GetConvolutionMethodString(m_ConvolutionMethod));
    if (descriptor.m_Parameters.m_BiasEnabled)
    {
        detailsInfo.m_BiasTensorInfo = armnn::Optional<armnn::TensorInfo>(descriptor.m_Bias->GetTensorInfo());
    }

    // Report Profiling Details
    ARMNN_REPORT_PROFILING_WORKLOAD_DESC("ClConvolution2dWorkload_Construct",
                                         descriptor.m_Parameters,
                                         detailsInfo,
                                         this->GetGuid());

    InitializeArmComputeClTensorData(*m_KernelTensor, m_Data.m_Weight);

    if (m_BiasTensor)
    {
        InitializeArmComputeClTensorData(*m_BiasTensor, m_Data.m_Bias);
    }

    // Force Compute Library to perform the necessary copying and reshaping, after which
    // delete all the input tensors that will no longer be needed
    {
        ARMNN_SCOPED_PROFILING_EVENT(Compute::Undefined, "ClConvolution2dWorkload_prepare");
        m_ConvolutionLayer.prepare();
    }
    FreeUnusedTensors();
}

void ClConvolution2dWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_CL_GUID("ClConvolution2dWorkload_Execute", this->GetGuid());
    RunClFunction(m_ConvolutionLayer, CHECK_LOCATION());
}

arm_compute::ConvolutionMethod ClConvolution2dWorkload::GetConvolutionMethod() const
{
    return m_ConvolutionMethod;
}

void ClConvolution2dWorkload::FreeUnusedTensors()
{
    FreeTensorIfUnused(m_KernelTensor);
    FreeTensorIfUnused(m_BiasTensor);
}

void ClConvolution2dWorkload::Reconfigure()
{
    arm_compute::ICLTensor& input  = static_cast<IClTensorHandle*>(m_Data.m_Inputs[0])->GetTensor();
    arm_compute::ICLTensor& output = static_cast<IClTensorHandle*>(m_Data.m_Outputs[0])->GetTensor();

    m_InputProxy->set(&input);
    m_OutputProxy->set(&output);
}

} //namespace armnn
