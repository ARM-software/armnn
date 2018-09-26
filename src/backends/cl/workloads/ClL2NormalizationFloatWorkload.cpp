//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ClL2NormalizationFloatWorkload.hpp"
#include <backends/cl/ClTensorHandle.hpp>
#include <backends/CpuTensorHandle.hpp>
#include <backends/aclCommon/ArmComputeUtils.hpp>

#include "ClWorkloadUtils.hpp"

namespace armnn
{
using namespace armcomputetensorutils;

arm_compute::Status ClL2NormalizationWorkloadValidate(const TensorInfo& input,
                                                      const TensorInfo& output,
                                                      const L2NormalizationDescriptor& descriptor)
{
    const arm_compute::TensorInfo aclInput = BuildArmComputeTensorInfo(input, descriptor.m_DataLayout);
    const arm_compute::TensorInfo aclOutput = BuildArmComputeTensorInfo(output, descriptor.m_DataLayout);

    arm_compute::NormalizationLayerInfo normalizationInfo =
            CreateAclNormalizationLayerInfoForL2Normalization(input);

    return arm_compute::CLNormalizationLayer::validate(&aclInput, &aclOutput, normalizationInfo);
}

ClL2NormalizationFloatWorkload::ClL2NormalizationFloatWorkload(const L2NormalizationQueueDescriptor& descriptor,
                                                               const WorkloadInfo& info)
    : FloatWorkload<L2NormalizationQueueDescriptor>(descriptor, info)
{
    m_Data.ValidateInputsOutputs("ClL2NormalizationFloatWorkload", 1, 1);

    arm_compute::ICLTensor& input  = static_cast<IClTensorHandle*>(m_Data.m_Inputs[0])->GetTensor();
    arm_compute::ICLTensor& output = static_cast<IClTensorHandle*>(m_Data.m_Outputs[0])->GetTensor();
    m_Layer.configure(&input, &output, CreateAclNormalizationLayerInfoForL2Normalization(info.m_InputTensorInfos[0]));
}

void ClL2NormalizationFloatWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_CL("ClL2NormalizationFloatWorkload_Execute");
    m_Layer.run();
}

} //namespace armnn



