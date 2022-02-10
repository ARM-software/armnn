//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ClNormalizationFloatWorkload.hpp"
#include <cl/ClTensorHandle.hpp>
#include <armnn/backends/TensorHandle.hpp>
#include <cl/ClLayerSupport.hpp>
#include <aclCommon/ArmComputeUtils.hpp>
#include <aclCommon/ArmComputeTensorUtils.hpp>
#include "ClWorkloadUtils.hpp"

using namespace armnn::armcomputetensorutils;

namespace armnn
{

arm_compute::Status ClNormalizationWorkloadValidate(const TensorInfo& input,
                                                    const TensorInfo& output,
                                                    const NormalizationDescriptor& descriptor)
{
    const arm_compute::TensorInfo aclInputInfo = BuildArmComputeTensorInfo(input, descriptor.m_DataLayout);
    const arm_compute::TensorInfo aclOutputInfo = BuildArmComputeTensorInfo(output, descriptor.m_DataLayout);

    arm_compute::NormalizationLayerInfo layerInfo = BuildArmComputeNormalizationLayerInfo(descriptor);

    return arm_compute::CLNormalizationLayer::validate(&aclInputInfo, &aclOutputInfo, layerInfo);
}

ClNormalizationFloatWorkload::ClNormalizationFloatWorkload(const NormalizationQueueDescriptor& descriptor,
                                                           const WorkloadInfo& info,
                                                           const arm_compute::CLCompileContext& clCompileContext)
    : FloatWorkload<NormalizationQueueDescriptor>(descriptor, info)
{
    // Report Profiling Details
    ARMNN_REPORT_PROFILING_WORKLOAD_DESC("ClNormalizationWorkload_Construct",
                                         descriptor.m_Parameters,
                                         info,
                                         this->GetGuid());

    m_Data.ValidateInputsOutputs("ClNormalizationFloatWorkload", 1, 1);

    arm_compute::ICLTensor& input  = static_cast<IClTensorHandle*>(m_Data.m_Inputs[0])->GetTensor();
    arm_compute::ICLTensor& output = static_cast<IClTensorHandle*>(m_Data.m_Outputs[0])->GetTensor();

    arm_compute::DataLayout aclDataLayout = ConvertDataLayout(m_Data.m_Parameters.m_DataLayout);
    input.info()->set_data_layout(aclDataLayout);
    output.info()->set_data_layout(aclDataLayout);

    arm_compute::NormalizationLayerInfo normalizationInfo = BuildArmComputeNormalizationLayerInfo(m_Data.m_Parameters);

    {
        ARMNN_SCOPED_PROFILING_EVENT(Compute::Undefined, "ClNormalizationFloatWorkload_configure");
        m_NormalizationLayer.configure(clCompileContext, &input, &output, normalizationInfo);
    }
};

void ClNormalizationFloatWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_CL_GUID("ClNormalizationFloatWorkload_Execute", this->GetGuid());
    RunClFunction(m_NormalizationLayer, CHECK_LOCATION());
}

void ClNormalizationFloatWorkload::ReplaceInputTensorHandle(ITensorHandle* tensorHandle, unsigned int slot)
{
    ITensorHandle* backupHandle = this->m_Data.m_Inputs[slot];
    this->m_Data.m_Inputs[slot] = tensorHandle;
    try
    {
        Reconfigure();
    }
    catch(armnn::UnimplementedException& e)
    {
        // Cannot reconfigure, revert the slot back and throw the exception.
        this->m_Data.m_Inputs[slot] = backupHandle;
        throw e;
    }
}

// Replace output tensor handle with the given TensorHandle
void ClNormalizationFloatWorkload::ReplaceOutputTensorHandle(ITensorHandle* tensorHandle, unsigned int slot)
{
    ITensorHandle* backupHandle = this->m_Data.m_Inputs[slot];
    this->m_Data.m_Inputs[slot] = tensorHandle;
    try
    {
        Reconfigure();
    }
    catch(armnn::UnimplementedException& e)
    {
        // Cannot reconfigure, revert the slot back and throw the exception.
        this->m_Data.m_Inputs[slot] = backupHandle;
        throw e;
    }
}

void ClNormalizationFloatWorkload::Reconfigure()
{
    throw armnn::UnimplementedException("Reconfigure not implemented for this workload");
}

} //namespace armnn
