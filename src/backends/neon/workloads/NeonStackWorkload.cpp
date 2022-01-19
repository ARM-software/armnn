//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include "NeonStackWorkload.hpp"
#include "NeonWorkloadUtils.hpp"

#include <aclCommon/ArmComputeTensorUtils.hpp>
#include <armnn/utility/NumericCast.hpp>
#include <armnn/utility/PolymorphicDowncast.hpp>
#include <armnn/backends/TensorHandle.hpp>
#include <neon/NeonTensorHandle.hpp>

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

arm_compute::Status NeonStackWorkloadValidate(const std::vector<const TensorInfo*>& inputs,
                                            const TensorInfo& output,
                                            const StackDescriptor& descriptor)
{
    std::vector<arm_compute::TensorInfo> aclInputs;
    for (const TensorInfo* input : inputs)
    {
        arm_compute::TensorInfo aclInputInfo = BuildArmComputeTensorInfo(*input, armnn::DataLayout::NCHW);
        aclInputs.emplace_back(aclInputInfo);
    }

    std::vector<arm_compute::ITensorInfo*> aclInputPtrs;
    for (arm_compute::ITensorInfo& input : aclInputs)
    {
        aclInputPtrs.emplace_back(&input);
    }

    const arm_compute::TensorInfo aclOutputInfo = BuildArmComputeTensorInfo(output);
    int aclAxis = CalcAxis(descriptor.m_Axis, descriptor.m_InputShape.GetNumDimensions());
    return arm_compute::NEStackLayer::validate(aclInputPtrs, aclAxis, &aclOutputInfo);
}

NeonStackWorkload::NeonStackWorkload(const StackQueueDescriptor& descriptor, const WorkloadInfo& info)
: NeonBaseWorkload<StackQueueDescriptor>(descriptor, info)
{
    // Report Profiling Details
    ARMNN_REPORT_PROFILING_WORKLOAD_DESC("NeonStackWorkload_Construct",
                                         descriptor.m_Parameters,
                                         info,
                                         this->GetGuid());

    std::vector<arm_compute::ITensor*> aclInputs;
    for (auto input : m_Data.m_Inputs)
    {
        arm_compute::ITensor& aclInput = PolymorphicPointerDowncast<IAclTensorHandle>(input)->GetTensor();
        aclInputs.emplace_back(&aclInput);
    }
    arm_compute::ITensor& output = PolymorphicPointerDowncast<IAclTensorHandle>(
        m_Data.m_Outputs[0])->GetTensor();

    m_Layer.reset(new arm_compute::NEStackLayer());
    int aclAxis = CalcAxis(descriptor.m_Parameters.m_Axis, descriptor.m_Parameters.m_InputShape.GetNumDimensions());
    m_Layer->configure(aclInputs, aclAxis, &output);
}

void NeonStackWorkload::Execute() const
{
    if (m_Layer)
    {
        ARMNN_SCOPED_PROFILING_EVENT_NEON_GUID("NeonStackWorkload_Execute", this->GetGuid());
        m_Layer->run();
    }
}

} //namespace armnn
