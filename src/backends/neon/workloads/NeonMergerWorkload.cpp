//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "NeonMergerWorkload.hpp"

#include "NeonWorkloadUtils.hpp"

#include <aclCommon/ArmComputeTensorUtils.hpp>
#include <backendsCommon/CpuTensorHandle.hpp>
#include <neon/NeonTensorHandle.hpp>

#include <arm_compute/runtime/NEON/functions/NEConcatenateLayer.h>

namespace armnn
{
using namespace armcomputetensorutils;

arm_compute::Status NeonMergerWorkloadValidate(const std::vector<const TensorInfo*>& inputs,
                                               const TensorInfo& output,
                                               const MergerDescriptor& descriptor)

{
    std::vector<arm_compute::TensorInfo> aclInputs;
    for (const TensorInfo* input : inputs)
    {
       arm_compute::TensorInfo aclInputInfo = BuildArmComputeTensorInfo(*input, armnn::DataLayout::NCHW);
       aclInputs.emplace_back(aclInputInfo);
    }
    const arm_compute::TensorInfo aclOutputInfo = BuildArmComputeTensorInfo(output);
    arm_compute::DataLayoutDimension aclAxis = arm_compute::DataLayoutDimension::WIDTH;

    std::vector<arm_compute::ITensorInfo*> aclInputPtrs;
    for (arm_compute::ITensorInfo& input : aclInputs)
    {
        aclInputPtrs.emplace_back(&input);
    }

    return arm_compute::NEConcatenateLayer::validate(aclInputPtrs, &aclOutputInfo, aclAxis);

}

NeonMergerWorkload::NeonMergerWorkload(
const MergerQueueDescriptor& descriptor, const WorkloadInfo& info)
        : BaseWorkload<MergerQueueDescriptor>(descriptor, info)
{
    m_Execute = true;

    unsigned int innerAxisOrder = descriptor.m_Parameters.GetNumDimensions() - descriptor.m_Parameters.GetConcatAxis();

    if (innerAxisOrder != 1)
    {
        m_Execute = false;
        return;
    }

    std::vector<arm_compute::ITensor *> aclInputs;
    arm_compute::DataLayout aclDataLayout = ConvertDataLayout(armnn::DataLayout::NCHW);
    for (auto input : m_Data.m_Inputs)
    {
        arm_compute::ITensor& aclInput  = boost::polymorphic_pointer_downcast<INeonTensorHandle>(input)->GetTensor();
        aclInput.info()->set_data_layout(aclDataLayout);
        aclInputs.emplace_back(&aclInput);
    }
    arm_compute::ITensor& output = boost::polymorphic_pointer_downcast<INeonTensorHandle>(
                                                                       m_Data.m_Outputs[0])->GetTensor();
    output.info()->set_data_layout(aclDataLayout);

    arm_compute::DataLayoutDimension aclAxis = arm_compute::DataLayoutDimension::WIDTH;

    auto layer = std::make_unique<arm_compute::NEConcatenateLayer>();
    layer->configure(aclInputs, &output, aclAxis);
    m_Layer.reset(layer.release());

    m_Layer->prepare();
}

void NeonMergerWorkload::Execute() const
{
    if (m_Execute)
    {
        ARMNN_SCOPED_PROFILING_EVENT_NEON("NeonMergerWorkload_Execute");
        m_Layer->run();
    }
}

} //namespace armnn

