//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "NeonStridedSliceWorkload.hpp"

#include "NeonWorkloadUtils.hpp"
#include <neon/NeonTensorHandle.hpp>
#include <aclCommon/ArmComputeUtils.hpp>
#include <aclCommon/ArmComputeTensorUtils.hpp>


namespace armnn
{

arm_compute::Status NeonStridedSliceWorkloadValidate(const TensorInfo& input,
                                                     const TensorInfo& output,
                                                     const StridedSliceDescriptor& descriptor)
{
    const arm_compute::TensorInfo aclInput = armcomputetensorutils::BuildArmComputeTensorInfo(input);
    const arm_compute::TensorInfo aclOutput = armcomputetensorutils::BuildArmComputeTensorInfo(output);

    arm_compute::Coordinates starts;
    arm_compute::Coordinates ends;
    arm_compute::Coordinates strides;

    std::tie(starts, ends, strides) = SetNeonStridedSliceData(descriptor.m_Begin,
                                                              descriptor.m_End,
                                                              descriptor.m_Stride);

    int32_t begin_mask = descriptor.m_BeginMask;
    int32_t end_mask = descriptor.m_EndMask;
    int32_t shrink_axis_mask = descriptor.m_ShrinkAxisMask;

    return arm_compute::NEStridedSlice::validate(&aclInput,
                                                 &aclOutput,
                                                 starts,
                                                 ends,
                                                 strides,
                                                 begin_mask,
                                                 end_mask,
                                                 shrink_axis_mask);
}

NeonStridedSliceWorkload::NeonStridedSliceWorkload(const StridedSliceQueueDescriptor& descriptor,
                                                   const WorkloadInfo& info)
        : BaseWorkload<StridedSliceQueueDescriptor>(descriptor, info)
{
    m_Data.ValidateInputsOutputs("NeonStridedSliceWorkload", 1, 1);

    arm_compute::ITensor& input = boost::polymorphic_downcast<IAclTensorHandle*>(m_Data.m_Inputs[0])->GetTensor();
    arm_compute::ITensor& output = boost::polymorphic_downcast<IAclTensorHandle*>(m_Data.m_Outputs[0])->GetTensor();

    arm_compute::Coordinates starts;
    arm_compute::Coordinates ends;
    arm_compute::Coordinates strides;

    std::tie(starts, ends, strides) = SetNeonStridedSliceData(m_Data.m_Parameters.m_Begin,
                                                              m_Data.m_Parameters.m_End,
                                                              m_Data.m_Parameters.m_Stride);

    int32_t begin_mask = m_Data.m_Parameters.m_BeginMask;
    int32_t end_mask = m_Data.m_Parameters.m_EndMask;
    int32_t shrink_axis_mask = m_Data.m_Parameters.m_ShrinkAxisMask;

    arm_compute::DataLayout aclDataLayout = ConvertDataLayout(m_Data.m_Parameters.m_DataLayout);
    input.info()->set_data_layout(aclDataLayout);
    output.info()->set_data_layout(aclDataLayout);

    auto layer = std::make_unique<arm_compute::NEStridedSlice>();

    layer->configure(&input,
                     &output,
                     starts,
                     ends,
                     strides,
                     begin_mask,
                     end_mask,
                     shrink_axis_mask);
    m_Layer.reset(layer.release());
}

void NeonStridedSliceWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_NEON("NeonStridedSliceWorkload_Execute");
    m_Layer->run();
}

} //namespace armnn