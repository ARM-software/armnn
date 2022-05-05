//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ClSqrtWorkload.hpp"

#include "ClWorkloadUtils.hpp"

#include <aclCommon/ArmComputeTensorUtils.hpp>
#include <aclCommon/ArmComputeUtils.hpp>
#include <armnn/utility/PolymorphicDowncast.hpp>

#include <cl/ClTensorHandle.hpp>

namespace armnn
{

arm_compute::Status ClSqrtWorkloadValidate(const TensorInfo& input, const TensorInfo& output)
{
    const arm_compute::TensorInfo aclInput  = armcomputetensorutils::BuildArmComputeTensorInfo(input);
    const arm_compute::TensorInfo aclOutput = armcomputetensorutils::BuildArmComputeTensorInfo(output);

    ActivationDescriptor descriptor;
    descriptor.m_Function = ActivationFunction::Sqrt;
    const arm_compute::ActivationLayerInfo activationLayerInfo =
            ConvertActivationDescriptorToAclActivationLayerInfo(descriptor);

    return arm_compute::CLActivationLayer::validate(&aclInput, &aclOutput, activationLayerInfo);
}

ClSqrtWorkload::ClSqrtWorkload(const ElementwiseUnaryQueueDescriptor& descriptor,
                               const WorkloadInfo& info,
                               const arm_compute::CLCompileContext& clCompileContext)
    : ClBaseWorkload<ElementwiseUnaryQueueDescriptor>(descriptor, info)
{
    ARMNN_ASSERT(descriptor.m_Parameters.m_Operation == UnaryOperation::Sqrt);

    // Report Profiling Details
    ARMNN_REPORT_PROFILING_WORKLOAD_DESC("ClSqrtWorkload_Construct",
                                         descriptor.m_Parameters,
                                         info,
                                         this->GetGuid());

    m_Data.ValidateInputsOutputs("ClSqrtWorkload", 1, 1);

    ActivationDescriptor activationDescriptor;
    activationDescriptor.m_Function = ActivationFunction::Sqrt;
    const arm_compute::ActivationLayerInfo activationLayerInfo =
            ConvertActivationDescriptorToAclActivationLayerInfo(activationDescriptor);

    arm_compute::ICLTensor& input  = PolymorphicDowncast<ClTensorHandle*>(m_Data.m_Inputs[0])->GetTensor();
    arm_compute::ICLTensor& output = PolymorphicDowncast<ClTensorHandle*>(m_Data.m_Outputs[0])->GetTensor();

    {
        ARMNN_SCOPED_PROFILING_EVENT(Compute::Undefined, "ClSqrtWorkload_configure");
        m_SqrtLayer.configure(clCompileContext, &input, &output, activationLayerInfo);
    }
}

void ClSqrtWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_CL_GUID("ClSqrtWorkload_Execute", this->GetGuid());
    RunClFunction(m_SqrtLayer, CHECK_LOCATION());
}

} // namespace armnn
