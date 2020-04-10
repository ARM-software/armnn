//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ClDequantizeWorkload.hpp"
#include "ClWorkloadUtils.hpp"

#include <aclCommon/ArmComputeTensorUtils.hpp>
#include <armnn/utility/PolymorphicDowncast.hpp>
#include <backendsCommon/CpuTensorHandle.hpp>

#include <arm_compute/core/Types.h>

#include <cl/ClLayerSupport.hpp>
#include <cl/ClTensorHandle.hpp>

namespace armnn
{
using namespace armcomputetensorutils;

arm_compute::Status ClDequantizeWorkloadValidate(const TensorInfo& input, const TensorInfo& output)
{
    const arm_compute::TensorInfo aclInputInfo = BuildArmComputeTensorInfo(input);
    const arm_compute::TensorInfo aclOutputInfo = BuildArmComputeTensorInfo(output);

    return arm_compute::CLDequantizationLayer::validate(&aclInputInfo, &aclOutputInfo);
}

ClDequantizeWorkload::ClDequantizeWorkload(const DequantizeQueueDescriptor& descriptor,
                                           const WorkloadInfo& workloadInfo)
                                           : BaseWorkload<DequantizeQueueDescriptor>(descriptor, workloadInfo)
{
    m_Data.ValidateInputsOutputs("ClDequantizeWorkload", 1, 1);

    arm_compute::ICLTensor& input = armnn::PolymorphicPointerDowncast<IClTensorHandle>(
            m_Data.m_Inputs[0])->GetTensor();

    arm_compute::ICLTensor& output = armnn::PolymorphicPointerDowncast<IClTensorHandle>(
            m_Data.m_Outputs[0])->GetTensor();

    m_Layer.reset(new arm_compute::CLDequantizationLayer());
    m_Layer->configure(&input, &output);
    m_Layer->prepare();
}

void ClDequantizeWorkload::Execute() const
{
    if (m_Layer)
    {
        ARMNN_SCOPED_PROFILING_EVENT_CL("ClDequantizeWorkload_Execute");
        m_Layer->run();
    }
}

} // namespace armnn
