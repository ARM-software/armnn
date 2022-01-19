//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include "ClStackWorkload.hpp"
#include "ClWorkloadUtils.hpp"
#include <aclCommon/ArmComputeTensorUtils.hpp>
#include <armnn/utility/NumericCast.hpp>
#include <armnn/utility/PolymorphicDowncast.hpp>
#include <armnn/backends/TensorHandle.hpp>
#include <cl/ClTensorHandle.hpp>
#include <cl/ClLayerSupport.hpp>

#include <arm_compute/core/Types.h>

namespace armnn
{
using namespace armcomputetensorutils;

namespace
{
int CalcAxis(const unsigned int axis, const unsigned int inputDimensions)
{
    const int intAxis = armnn::numeric_cast<int>(axis);
    return armnn::numeric_cast<int>(inputDimensions) - intAxis;
}
} //namespace

arm_compute::Status ClStackWorkloadValidate(const std::vector<const TensorInfo*>& inputs,
                                            const TensorInfo& output,
                                            const StackDescriptor& descriptor)
{
    std::vector<arm_compute::ITensorInfo*> aclInputPtrs;
    arm_compute::TensorInfo aclInputInfo;
    for (const TensorInfo* input : inputs)
    {
        aclInputInfo = BuildArmComputeTensorInfo(*input);
        aclInputPtrs.emplace_back(&aclInputInfo);
    }
    const arm_compute::TensorInfo aclOutputInfo = BuildArmComputeTensorInfo(output);

    int aclAxis = CalcAxis(descriptor.m_Axis, descriptor.m_InputShape.GetNumDimensions());

    return arm_compute::CLStackLayer::validate(aclInputPtrs, aclAxis, &aclOutputInfo);
}

ClStackWorkload::ClStackWorkload(const StackQueueDescriptor& descriptor,
                                 const WorkloadInfo& info,
                                 const arm_compute::CLCompileContext& clCompileContext)
: ClBaseWorkload<StackQueueDescriptor>(descriptor, info)
{
    // Report Profiling Details
    ARMNN_REPORT_PROFILING_WORKLOAD_DESC("ClStackWorkload_Construct",
                                         descriptor.m_Parameters,
                                         info,
                                         this->GetGuid());

    std::vector<arm_compute::ICLTensor*> aclInputs;
    for (auto input : m_Data.m_Inputs)
    {
        arm_compute::ICLTensor& aclInput = armnn::PolymorphicPointerDowncast<IClTensorHandle>(input)->GetTensor();
        aclInputs.emplace_back(&aclInput);
    }
    arm_compute::ICLTensor& output = armnn::PolymorphicPointerDowncast<IClTensorHandle>(
                                                                         m_Data.m_Outputs[0])->GetTensor();

    m_Layer.reset(new arm_compute::CLStackLayer());
    int aclAxis = CalcAxis(descriptor.m_Parameters.m_Axis, descriptor.m_Parameters.m_InputShape.GetNumDimensions());
    {
        ARMNN_SCOPED_PROFILING_EVENT(Compute::Undefined, "ClStackWorkload_configure");
        m_Layer->configure(clCompileContext, aclInputs, aclAxis, &output);
    }
}

void ClStackWorkload::Execute() const
{
    if (m_Layer)
    {
        ARMNN_SCOPED_PROFILING_EVENT_CL_GUID("ClStackWorkload_Execute", this->GetGuid());
        m_Layer->run();
    }
}

} //namespace armnn