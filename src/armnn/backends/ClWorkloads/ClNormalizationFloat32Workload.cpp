//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//

#include "ClNormalizationFloat32Workload.hpp"
#include "backends/ClTensorHandle.hpp"
#include "backends/CpuTensorHandle.hpp"
#include "backends/ClLayerSupport.hpp"
#include "backends/ArmComputeUtils.hpp"
#include "backends/ArmComputeTensorUtils.hpp"

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

ClNormalizationFloat32Workload::ClNormalizationFloat32Workload(const NormalizationQueueDescriptor& descriptor,
                                                               const WorkloadInfo& info)
    : Float32Workload<NormalizationQueueDescriptor>(descriptor, info)
{
    m_Data.ValidateInputsOutputs("ClNormalizationFloat32Workload", 1, 1);

    arm_compute::ICLTensor& input  = static_cast<IClTensorHandle*>(m_Data.m_Inputs[0])->GetTensor();
    arm_compute::ICLTensor& output = static_cast<IClTensorHandle*>(m_Data.m_Outputs[0])->GetTensor();

    arm_compute::NormalizationLayerInfo normalizationInfo =
        armcomputetensorutils::BuildArmComputeNormalizationLayerInfo(m_Data.m_Parameters);

    m_NormalizationLayer.configure(&input, &output, normalizationInfo);
};

void ClNormalizationFloat32Workload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT(Compute::GpuAcc, "ClNormalizationFloat32Workload_Execute");
    m_NormalizationLayer.run();
}

} //namespace armnn