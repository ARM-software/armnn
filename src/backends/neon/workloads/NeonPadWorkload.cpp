//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "NeonPadWorkload.hpp"

#include <neon/NeonTensorHandle.hpp>
#include <aclCommon/ArmComputeTensorUtils.hpp>
#include <arm_compute/core/Types.h>
#include <arm_compute/runtime/NEON/functions/NEPadLayer.h>

#include "NeonWorkloadUtils.hpp"

namespace armnn
{
using namespace armcomputetensorutils;

NeonPadWorkload::NeonPadWorkload(const PadQueueDescriptor& descriptor, const WorkloadInfo& info)
    : BaseWorkload<PadQueueDescriptor>(descriptor, info)
{
    m_Data.ValidateInputsOutputs("NeonPadWorkload", 1, 1);

    arm_compute::ITensor& input = static_cast<INeonTensorHandle*>(m_Data.m_Inputs[0])->GetTensor();
    arm_compute::ITensor& output = static_cast<INeonTensorHandle*>(m_Data.m_Outputs[0])->GetTensor();

    std::vector<std::pair<unsigned int, unsigned int>> reversed_PadList(descriptor.m_Parameters.m_PadList.size());

    std::reverse_copy(std::begin(descriptor.m_Parameters.m_PadList),
                      std::end(descriptor.m_Parameters.m_PadList),
                      std::begin(reversed_PadList));

    arm_compute::PaddingList padList = static_cast<arm_compute::PaddingList>(reversed_PadList);

    auto layer = std::make_unique<arm_compute::NEPadLayer>();
    layer->configure(&input, &output, padList);
    m_Layer.reset(layer.release());
}

void NeonPadWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_NEON("NeonPadWorkload_Execute");
    m_Layer->run();
}

arm_compute::Status NeonPadWorkloadValidate(const TensorInfo& input,
                                            const TensorInfo& output,
                                            const PadDescriptor& descriptor)
{
    const arm_compute::TensorInfo aclInputInfo = BuildArmComputeTensorInfo(input);
    const arm_compute::TensorInfo aclOutputInfo = BuildArmComputeTensorInfo(output);
    arm_compute::PaddingList padList = static_cast<arm_compute::PaddingList>(descriptor.m_PadList);

    return arm_compute::NEPadLayer::validate(&aclInputInfo, &aclOutputInfo, padList);
}

} // namespace armnn
