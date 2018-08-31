//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//

#include "ClL2NormalizationFloat32Workload.hpp"
#include "backends/ClTensorHandle.hpp"
#include "backends/CpuTensorHandle.hpp"
#include "backends/ArmComputeUtils.hpp"

namespace armnn
{
using namespace armcomputetensorutils;

arm_compute::Status ClL2NormalizationWorkloadValidate(const TensorInfo& input,
                                                      const TensorInfo& output)
{
    const arm_compute::TensorInfo aclInput = BuildArmComputeTensorInfo(input);
    const arm_compute::TensorInfo aclOutput = BuildArmComputeTensorInfo(output);

    arm_compute::NormalizationLayerInfo normalizationInfo =
            CreateAclNormalizationLayerInfoForL2Normalization(input);

    return arm_compute::CLNormalizationLayer::validate(&aclInput, &aclOutput, normalizationInfo);
}

ClL2NormalizationFloat32Workload::ClL2NormalizationFloat32Workload(const L2NormalizationQueueDescriptor& descriptor,
                                                                   const WorkloadInfo& info)
    : FloatWorkload<L2NormalizationQueueDescriptor>(descriptor, info)
{
    m_Data.ValidateInputsOutputs("ClL2NormalizationFloat32Workload", 1, 1);

    arm_compute::ICLTensor& input  = static_cast<IClTensorHandle*>(m_Data.m_Inputs[0])->GetTensor();
    arm_compute::ICLTensor& output = static_cast<IClTensorHandle*>(m_Data.m_Outputs[0])->GetTensor();
    m_Layer.configure(&input, &output, CreateAclNormalizationLayerInfoForL2Normalization(info.m_InputTensorInfos[0]));
}

void ClL2NormalizationFloat32Workload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_CL("ClL2NormalizationFloat32Workload_Execute");
    m_Layer.run();
}

} //namespace armnn



