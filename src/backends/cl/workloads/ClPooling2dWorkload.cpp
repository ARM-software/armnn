//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ClPooling2dWorkload.hpp"
#include <cl/ClLayerSupport.hpp>
#include <cl/ClTensorHandle.hpp>
#include <aclCommon/ArmComputeUtils.hpp>
#include <aclCommon/ArmComputeTensorUtils.hpp>

#include "ClWorkloadUtils.hpp"

namespace armnn
{
using namespace armcomputetensorutils;

arm_compute::Status ClPooling2dWorkloadValidate(const TensorInfo& input,
    const TensorInfo& output,
    const Pooling2dDescriptor& descriptor)
{
    const arm_compute::TensorInfo aclInputInfo = BuildArmComputeTensorInfo(input, descriptor.m_DataLayout);
    const arm_compute::TensorInfo aclOutputInfo = BuildArmComputeTensorInfo(output, descriptor.m_DataLayout);

    arm_compute::PoolingLayerInfo layerInfo = BuildArmComputePoolingLayerInfo(descriptor);

    return arm_compute::CLPoolingLayer::validate(&aclInputInfo, &aclOutputInfo, layerInfo);
}

ClPooling2dWorkload::ClPooling2dWorkload(
    const Pooling2dQueueDescriptor& descriptor,
    const WorkloadInfo& info,
    const arm_compute::CLCompileContext& clCompileContext)
    : ClBaseWorkload<Pooling2dQueueDescriptor>(descriptor, info)
{
    // Report Profiling Details
    ARMNN_REPORT_PROFILING_WORKLOAD_DESC("ClPooling2dWorkload_Construct",
                                         descriptor.m_Parameters,
                                         info,
                                         this->GetGuid());

    m_Data.ValidateInputsOutputs("ClPooling2dWorkload", 1, 1);

    arm_compute::ICLTensor& input = static_cast<IClTensorHandle*>(m_Data.m_Inputs[0])->GetTensor();
    arm_compute::ICLTensor& output = static_cast<IClTensorHandle*>(m_Data.m_Outputs[0])->GetTensor();

    arm_compute::DataLayout aclDataLayout = ConvertDataLayout(m_Data.m_Parameters.m_DataLayout);
    input.info()->set_data_layout(aclDataLayout);
    output.info()->set_data_layout(aclDataLayout);

    // flag to use wider accumulators (32 bit instead of 16 for FP16) to improve accuracy
    // enable fp_mixed_precision for the the FP16 cases that
    // accumulation reaches a limit beyond which there is no more increment of the value
    bool fpMixedPrecision = false;

    arm_compute::PoolingLayerInfo layerInfo = BuildArmComputePoolingLayerInfo(m_Data.m_Parameters, fpMixedPrecision);

    {
        ARMNN_SCOPED_PROFILING_EVENT(Compute::Undefined, "ClPooling2dWorkload_configure");
        // Run the layer.
        m_PoolingLayer.configure(clCompileContext, &input, &output, layerInfo);
    }
}

void ClPooling2dWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_CL_GUID("ClPooling2dWorkload_Execute", this->GetGuid());
    RunClFunction(m_PoolingLayer, CHECK_LOCATION());
}

}
