//
// Copyright © 2022-2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ClBatchMatMulWorkload.hpp"

#include "ClWorkloadUtils.hpp"

#include <aclCommon/ArmComputeTensorUtils.hpp>
#include <aclCommon/ArmComputeUtils.hpp>

#include <armnn/utility/PolymorphicDowncast.hpp>

#include <backendsCommon/WorkloadUtils.hpp>

#include <cl/ClTensorHandle.hpp>

#include <arm_compute/core/MatMulInfo.h>

namespace armnn
{

arm_compute::Status ClBatchMatMulValidate(const TensorInfo& inputInfoX,
                                          const TensorInfo& inputInfoY,
                                          const TensorInfo& outputInfo,
                                          const BatchMatMulDescriptor& descriptor,
                                          const ActivationDescriptor* activationDescriptor)
{
    if (descriptor.m_AdjointX || descriptor.m_AdjointY )
    {
        throw Exception("Support for adjoint not implemented.");
    }
    if (descriptor.m_DataLayoutX != armnn::DataLayout::NCHW || descriptor.m_DataLayoutY != armnn::DataLayout::NCHW )
    {
        throw Exception("Only supported the MatMul in the last 2 dimensions");
    }

    arm_compute::TensorInfo aclInputInfoX = armcomputetensorutils::BuildArmComputeTensorInfo(inputInfoX);
    arm_compute::TensorInfo aclInputInfoY = armcomputetensorutils::BuildArmComputeTensorInfo(inputInfoY);
    const arm_compute::TensorInfo aclOutputInfo = armcomputetensorutils::BuildArmComputeTensorInfo(outputInfo);

    // GeMM dispatches kernel handles dynamic inputs differently to static so this flag needs to be set
    aclInputInfoX.set_are_values_constant(false);
    aclInputInfoY.set_are_values_constant(false);

    const arm_compute::ActivationLayerInfo activationInfo = ConvertActivationDescriptorToAclActivationLayerInfo(
            activationDescriptor);

    arm_compute::MatMulInfo matMulInfo;
    matMulInfo.adj_lhs(descriptor.m_TransposeX);
    matMulInfo.adj_rhs(descriptor.m_TransposeY);
    matMulInfo.fused_activation(activationInfo);

    return arm_compute::CLMatMul::validate(&aclInputInfoX, &aclInputInfoY, &aclOutputInfo, matMulInfo);
}

ClBatchMatMulWorkload::ClBatchMatMulWorkload(const BatchMatMulQueueDescriptor& descriptor,
                                             const WorkloadInfo& info,
                                             const arm_compute::CLCompileContext& clCompileContext)
    : ClBaseWorkload<BatchMatMulQueueDescriptor>(descriptor, info)
{
    // Report Profiling Details
    ARMNN_REPORT_PROFILING_WORKLOAD_DESC("ClBatchMatMulWorkload_Construct",
                                         descriptor.m_Parameters,
                                         info,
                                         this->GetGuid());

    if (descriptor.m_Parameters.m_AdjointX || descriptor.m_Parameters.m_AdjointY )
    {
        throw Exception("Support for adjoint not implemented.");
    }
    if (descriptor.m_Parameters.m_DataLayoutX != armnn::DataLayout::NCHW ||
        descriptor.m_Parameters.m_DataLayoutY != armnn::DataLayout::NCHW )
    {
        throw Exception("Only supported the MatMul in the last 2 dimensions");
    }

    m_Data.ValidateInputsOutputs("ClBatchMatMulWorkload", 2, 1);

    arm_compute::ICLTensor& inputX = PolymorphicDowncast<ClTensorHandle*>(m_Data.m_Inputs[0])->GetTensor();
    arm_compute::ICLTensor& inputY = PolymorphicDowncast<ClTensorHandle*>(m_Data.m_Inputs[1])->GetTensor();
    auto outputHandle = PolymorphicDowncast<ClTensorHandle*>(m_Data.m_Outputs[0]);
    arm_compute::ICLTensor& output = outputHandle->GetTensor();

    // GeMM dispatches kernel handles dynamic inputs differently to static so this flag needs to be set
    inputX.info()->set_are_values_constant(false);
    inputY.info()->set_are_values_constant(false);

    const arm_compute::ActivationLayerInfo activationInfo = ConvertAdditionalInfoToAclActivationLayerInfo(descriptor);

    arm_compute::MatMulInfo matMulInfo;
    matMulInfo.adj_lhs(descriptor.m_Parameters.m_TransposeX);
    matMulInfo.adj_rhs(descriptor.m_Parameters.m_TransposeY);
    matMulInfo.fused_activation(activationInfo);

    m_MatMulLayer.configure(clCompileContext, &inputX, &inputY, &output, matMulInfo);

    // Report Profiling Details
    WorkloadInfo detailsInfo;
    detailsInfo.m_InputTensorInfos = info.m_InputTensorInfos;
    detailsInfo.m_OutputTensorInfos = info.m_OutputTensorInfos;
    ARMNN_REPORT_PROFILING_WORKLOAD_DESC("ClBatchMatMulWorkload_Construct",
                                         descriptor.m_Parameters,
                                         detailsInfo,
                                         GetGuid());
}

void ClBatchMatMulWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_CL_GUID("ClBatchMatMulWorkload_Execute", this->GetGuid());
    RunClFunction(m_MatMulLayer, CHECK_LOCATION());
}
} //namespace armnn
