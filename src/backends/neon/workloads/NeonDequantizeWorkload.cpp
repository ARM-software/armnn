//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "NeonDequantizeWorkload.hpp"

#include "NeonWorkloadUtils.hpp"

#include <arm_compute/runtime/NEON/functions/NEDequantizationLayer.h>

#include <aclCommon/ArmComputeTensorUtils.hpp>
#include <armnn/utility/PolymorphicDowncast.hpp>
#include <armnn/backends/TensorHandle.hpp>
#include <neon/NeonTensorHandle.hpp>

namespace armnn
{

using namespace armcomputetensorutils;

arm_compute::Status NeonDequantizeWorkloadValidate(const TensorInfo& input,
                                                   const TensorInfo& output)
{
    const arm_compute::TensorInfo aclInput = BuildArmComputeTensorInfo(input);
    const arm_compute::TensorInfo aclOutput = BuildArmComputeTensorInfo(output);

    return arm_compute::NEDequantizationLayer::validate(&aclInput, &aclOutput);
}

NeonDequantizeWorkload::NeonDequantizeWorkload(const DequantizeQueueDescriptor& descriptor, const WorkloadInfo& info)
        : NeonBaseWorkload<DequantizeQueueDescriptor>(descriptor, info)
{
    m_Data.ValidateInputsOutputs("NeonDequantizeWorkload", 1, 1);

    arm_compute::ITensor& input = PolymorphicDowncast<IAclTensorHandle*>(m_Data.m_Inputs[0])->GetTensor();
    arm_compute::ITensor& output = PolymorphicDowncast<IAclTensorHandle*>(m_Data.m_Outputs[0])->GetTensor();

    std::unique_ptr<arm_compute::NEDequantizationLayer> layer(new arm_compute::NEDequantizationLayer());
    layer->configure(&input, &output);
    layer->prepare();
    m_Layer.reset(layer.release());
}

void NeonDequantizeWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_NEON_GUID("NeonDequantizeWorkload_Execute", this->GetGuid());
    m_Layer->run();
}

} //namespace armnn

