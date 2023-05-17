//
// Copyright © 2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ClElementwiseBinaryWorkload.hpp"

#include <cl/ClTensorHandle.hpp>
#include <armnn/backends/TensorHandle.hpp>
#include <aclCommon/ArmComputeTensorUtils.hpp>
#include <aclCommon/ArmComputeUtils.hpp>

#include "ClWorkloadUtils.hpp"

namespace armnn
{
using namespace armcomputetensorutils;

ClElementwiseBinaryWorkload::ClElementwiseBinaryWorkload(const ElementwiseBinaryQueueDescriptor& descriptor,
                                                         const WorkloadInfo& info,
                                                         const arm_compute::CLCompileContext& clCompileContext)
    : ClBaseWorkload<ElementwiseBinaryQueueDescriptor>(descriptor, info)
{
    this->m_Data.ValidateInputsOutputs("ClElementwiseBinaryWorkload", 2, 1);

    arm_compute::ICLTensor &input0 = static_cast<IClTensorHandle *>(this->m_Data.m_Inputs[0])->GetTensor();
    arm_compute::ICLTensor &input1 = static_cast<IClTensorHandle *>(this->m_Data.m_Inputs[1])->GetTensor();
    arm_compute::ICLTensor &output = static_cast<IClTensorHandle *>(this->m_Data.m_Outputs[0])->GetTensor();

    const arm_compute::ActivationLayerInfo activationInfo = ConvertAdditionalInfoToAclActivationLayerInfo(descriptor);
    {
        ARMNN_SCOPED_PROFILING_EVENT(Compute::Undefined, "ClElementwiseBinaryWorkload_configure");

        switch (descriptor.m_Parameters.m_Operation)
        {
            case armnn::BinaryOperation::Power:
            {
                auto powerLayer = std::make_unique<arm_compute::CLElementwisePower>();
                powerLayer->configure(clCompileContext, &input0, &input1, &output, activationInfo);
                m_ElementwiseBinaryLayer.reset(powerLayer.release());
                break;
            }
            case armnn::BinaryOperation::SqDiff:
            {
                auto SqDiffLayer = std::make_unique<arm_compute::CLElementwiseSquaredDiff>();
                SqDiffLayer->configure(clCompileContext, &input0, &input1, &output, activationInfo);
                m_ElementwiseBinaryLayer.reset(SqDiffLayer.release());
                break;
            }
            default:
                throw InvalidArgumentException("Unknown binary operator", CHECK_LOCATION());
        }
    }
}
void ClElementwiseBinaryWorkload::Execute() const
{
    if (m_ElementwiseBinaryLayer)
    {
        ARMNN_SCOPED_PROFILING_EVENT_CL_GUID("ClElementwiseBinaryWorkload_Execute", this->GetGuid());
        m_ElementwiseBinaryLayer->run();
    }
}

arm_compute::Status ClElementwiseBinaryValidate(const TensorInfo& input0,
                                                const TensorInfo& input1,
                                                const TensorInfo& output,
                                                const ElementwiseBinaryDescriptor& descriptor,
                                                const ActivationDescriptor* activationDescriptor)
{
    const arm_compute::TensorInfo aclInput0Info = BuildArmComputeTensorInfo(input0);
    const arm_compute::TensorInfo aclInput1Info = BuildArmComputeTensorInfo(input1);
    const arm_compute::TensorInfo aclOutputInfo = BuildArmComputeTensorInfo(output);

    const arm_compute::ActivationLayerInfo activationInfo = ConvertActivationDescriptorToAclActivationLayerInfo(
            activationDescriptor);

    switch (descriptor.m_Operation)
    {
        case armnn::BinaryOperation::Power:
            return arm_compute::CLElementwisePower::validate(&aclInput0Info,
                                                             &aclInput1Info,
                                                             &aclOutputInfo,
                                                             activationInfo);
        case armnn::BinaryOperation::SqDiff:
            return arm_compute::CLElementwiseSquaredDiff::validate(&aclInput0Info,
                                                                   &aclInput1Info,
                                                                   &aclOutputInfo,
                                                                   activationInfo);
        default:
            throw InvalidArgumentException("Unknown binary operator", CHECK_LOCATION());
    }
}

} //namespace armnn