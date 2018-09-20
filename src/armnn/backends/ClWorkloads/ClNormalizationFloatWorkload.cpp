//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ClNormalizationFloatWorkload.hpp"
#include "backends/ClTensorHandle.hpp"
#include "backends/CpuTensorHandle.hpp"
#include "backends/ClLayerSupport.hpp"
#include "backends/ArmComputeUtils.hpp"
#include "backends/ArmComputeTensorUtils.hpp"
#include "ClWorkloadUtils.hpp"

namespace armnn
{

arm_compute::Status ClNormalizationWorkloadValidate(const TensorInfo& input, const TensorInfo& output,
    const NormalizationDescriptor& descriptor)
{
    const arm_compute::TensorInfo aclInputInfo = armcomputetensorutils::BuildArmComputeTensorInfo(input);
    const arm_compute::TensorInfo aclOutputInfo = armcomputetensorutils::BuildArmComputeTensorInfo(output);

    arm_compute::NormalizationLayerInfo layerInfo =
        armcomputetensorutils::BuildArmComputeNormalizationLayerInfo(descriptor);

    return arm_compute::CLNormalizationLayer::validate(&aclInputInfo, &aclOutputInfo, layerInfo);
}

ClNormalizationFloatWorkload::ClNormalizationFloatWorkload(const NormalizationQueueDescriptor& descriptor,
                                                           const WorkloadInfo& info)
    : FloatWorkload<NormalizationQueueDescriptor>(descriptor, info)
{
    m_Data.ValidateInputsOutputs("ClNormalizationFloatWorkload", 1, 1);

    arm_compute::ICLTensor& input  = static_cast<IClTensorHandle*>(m_Data.m_Inputs[0])->GetTensor();
    arm_compute::ICLTensor& output = static_cast<IClTensorHandle*>(m_Data.m_Outputs[0])->GetTensor();

    arm_compute::NormalizationLayerInfo normalizationInfo =
        armcomputetensorutils::BuildArmComputeNormalizationLayerInfo(m_Data.m_Parameters);

    m_NormalizationLayer.configure(&input, &output, normalizationInfo);
};

void ClNormalizationFloatWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_CL("ClNormalizationFloatWorkload_Execute");
    m_NormalizationLayer.run();
}

} //namespace armnn
