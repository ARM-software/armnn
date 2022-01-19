//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ClConvolution3dWorkload.hpp"

#include "ClWorkloadUtils.hpp"

#include <cl/ClLayerSupport.hpp>
#include <cl/ClTensorHandle.hpp>
#include <cl/ClLayerSupport.hpp>
#include <aclCommon/ArmComputeUtils.hpp>
#include <aclCommon/ArmComputeTensorUtils.hpp>
#include <armnn/backends/TensorHandle.hpp>

#include <arm_compute/runtime/CL/functions/CLConv3D.h>

namespace armnn
{
using namespace armcomputetensorutils;

arm_compute::Status ClConvolution3dWorkloadValidate(const TensorInfo& input,
                                                    const TensorInfo& output,
                                                    const Convolution3dDescriptor& descriptor,
                                                    const TensorInfo& weights,
                                                    const Optional<TensorInfo>& biases,
                                                    bool isFastMathEnabled,
                                                    const ActivationDescriptor* activationDescriptor)
{
    const arm_compute::TensorInfo aclInputInfo = BuildArmComputeTensorInfo(input, descriptor.m_DataLayout);
    const arm_compute::TensorInfo aclWeightsInfo = BuildArmComputeTensorInfo(weights, descriptor.m_DataLayout);

    arm_compute::TensorInfo aclBiasesInfo;
    arm_compute::TensorInfo* optionalAclBiasesInfo = nullptr;
    if (descriptor.m_BiasEnabled)
    {
        ARMNN_ASSERT(biases.has_value());
        aclBiasesInfo = BuildArmComputeTensorInfo(biases.value(), descriptor.m_DataLayout);
        optionalAclBiasesInfo = &aclBiasesInfo;
    }

    const arm_compute::TensorInfo aclOutputInfo = BuildArmComputeTensorInfo(output, descriptor.m_DataLayout);

    const arm_compute::Conv3dInfo aclConv3DInfo = ComputeConv3DInfo(descriptor,
                                                                    isFastMathEnabled,
                                                                    activationDescriptor);

    return arm_compute::CLConv3D::validate(&aclInputInfo,
                                           &aclWeightsInfo,
                                           optionalAclBiasesInfo,
                                           &aclOutputInfo,
                                           aclConv3DInfo);
}

ClConvolution3dWorkload::ClConvolution3dWorkload(const Convolution3dQueueDescriptor& descriptor,
                                                 const WorkloadInfo& info,
                                                 std::shared_ptr<arm_compute::MemoryManagerOnDemand>& memoryManager,
                                                 const arm_compute::CLCompileContext& clCompileContext,
                                                 const bool isFastMathEnabled)
    : ClBaseWorkload<Convolution3dQueueDescriptor>(descriptor, info)
    , m_ConvolutionLayer()
{
    IgnoreUnused(memoryManager);

    uint32_t numInputs = m_Data.m_Parameters.m_BiasEnabled ? 3: 2;
    m_Data.ValidateInputsOutputs("ClConvolution3dWorkload", numInputs, 1);

    arm_compute::ICLTensor& input  = static_cast<IClTensorHandle*>(m_Data.m_Inputs[0])->GetTensor();
    arm_compute::ICLTensor& weights  = static_cast<IClTensorHandle*>(m_Data.m_Inputs[1])->GetTensor();
    arm_compute::ICLTensor* biasesPtr = nullptr;
    if (m_Data.m_Parameters.m_BiasEnabled)
    {
        biasesPtr = &static_cast<IClTensorHandle*>(m_Data.m_Inputs[2])->GetTensor();
    }
    arm_compute::ICLTensor& output = static_cast<IClTensorHandle*>(m_Data.m_Outputs[0])->GetTensor();

    arm_compute::DataLayout aclDataLayout = ConvertDataLayout(m_Data.m_Parameters.m_DataLayout);
    input.info()->set_data_layout(aclDataLayout);
    weights.info()->set_data_layout(aclDataLayout);
    output.info()->set_data_layout(aclDataLayout);

    const arm_compute::Conv3dInfo aclConv3DInfo = ComputeConv3DInfo(descriptor,
                                                                    isFastMathEnabled);

    {
        ARMNN_SCOPED_PROFILING_EVENT(Compute::Undefined, "ClConvolution3dWorkload_configure");
        m_ConvolutionLayer.configure(clCompileContext,
                                     &input,
                                     &weights,
                                     biasesPtr,
                                     &output,
                                     aclConv3DInfo);
    }
     // Add details for profiling output
    WorkloadInfo detailsInfo;

    detailsInfo.m_InputTensorInfos = info.m_InputTensorInfos;
    detailsInfo.m_OutputTensorInfos = info.m_OutputTensorInfos;

    // Report Profiling Details
    ARMNN_REPORT_PROFILING_WORKLOAD_DESC("ClConvolution3dWorkload_Construct",
                                         descriptor.m_Parameters,
                                         detailsInfo,
                                         this->GetGuid());

    // Force Compute Library to perform the necessary copying and reshaping, after which
    // delete all the input tensors that will no longer be needed
    m_ConvolutionLayer.prepare();
}

void ClConvolution3dWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_CL_GUID("ClConvolution3dWorkload_Execute", this->GetGuid());
    RunClFunction(m_ConvolutionLayer, CHECK_LOCATION());
}

} //namespace armnn
