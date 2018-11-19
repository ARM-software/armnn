//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include "ClMergerWorkload.hpp"
#include "ClWorkloadUtils.hpp"
#include <aclCommon/ArmComputeTensorUtils.hpp>
#include <backendsCommon/CpuTensorHandle.hpp>
#include <cl/ClTensorHandle.hpp>
#include <cl/ClLayerSupport.hpp>

#include <boost/polymorphic_pointer_cast.hpp>

namespace armnn
{
using namespace armcomputetensorutils;

arm_compute::Status ClMergerWorkloadValidate(const std::vector<const TensorInfo*>& inputs,
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

    return arm_compute::CLConcatenateLayer::validate(aclInputPtrs, &aclOutputInfo, aclAxis);

}

ClMergerWorkload::ClMergerWorkload(const MergerQueueDescriptor& descriptor, const WorkloadInfo& info)
: BaseWorkload<MergerQueueDescriptor>(descriptor, info)
{
    m_Execute = true;

    unsigned int innerAxisOrder = descriptor.m_Parameters.GetNumDimensions() - descriptor.m_Parameters.GetConcatAxis();

    if (innerAxisOrder != 1)
    {
        m_Execute = false;
        return;
    }

    std::vector<arm_compute::ICLTensor *> aclInputs;
    arm_compute::DataLayout aclDataLayout = ConvertDataLayout(armnn::DataLayout::NCHW);
    for (auto input : m_Data.m_Inputs)
    {
        arm_compute::ICLTensor& aclInput  = boost::polymorphic_pointer_downcast<IClTensorHandle>(input)->GetTensor();
        aclInput.info()->set_data_layout(aclDataLayout);
        aclInputs.emplace_back(&aclInput);
    }
    arm_compute::ICLTensor& output = boost::polymorphic_pointer_downcast<IClTensorHandle>(
                                                                         m_Data.m_Outputs[0])->GetTensor();
    output.info()->set_data_layout(aclDataLayout);

    arm_compute::DataLayoutDimension aclAxis = arm_compute::DataLayoutDimension::WIDTH;

    m_Layer.configure(aclInputs, &output, aclAxis);

    m_Layer.prepare();

}

void ClMergerWorkload::Execute() const
{
    if (m_Execute)
    {
        ARMNN_SCOPED_PROFILING_EVENT_CL("ClMergerWorkload_Execute");
        m_Layer.run();
    }

}

} //namespace armnn