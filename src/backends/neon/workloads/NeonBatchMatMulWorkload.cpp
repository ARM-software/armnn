//
// Copyright © 2022-2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "NeonBatchMatMulWorkload.hpp"

#include "NeonWorkloadUtils.hpp"

#include <armnn/utility/PolymorphicDowncast.hpp>
#include <aclCommon/ArmComputeUtils.hpp>

#include <backendsCommon/WorkloadUtils.hpp>

#include <arm_compute/function_info/MatMulInfo.h>

namespace armnn
{
arm_compute::Status NeonBatchMatMulValidate(const TensorInfo& inputInfoX,
                                            const TensorInfo& inputInfoY,
                                            const TensorInfo& outputInfo,
                                            const BatchMatMulDescriptor& descriptor,
                                            const bool isFastMathEnabled,
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
    arm_compute::TensorInfo aclOutputInfo = armcomputetensorutils::BuildArmComputeTensorInfo(outputInfo);

    // GeMM dispatches kernel handles dynamic inputs differently to static so this flag needs to be set
    aclInputInfoX.set_are_values_constant(false);
    aclInputInfoY.set_are_values_constant(false);

    const arm_compute::ActivationLayerInfo activationInfo = ConvertActivationDescriptorToAclActivationLayerInfo(
            activationDescriptor);

    arm_compute::MatMulInfo matMulInfo;
    matMulInfo.adj_lhs(descriptor.m_TransposeX);
    matMulInfo.adj_rhs(descriptor.m_TransposeY);

    arm_compute::CpuMatMulSettings settings;
    settings.fast_math(isFastMathEnabled);

    return arm_compute::NEMatMul::validate(&aclInputInfoX, &aclInputInfoY, &aclOutputInfo, matMulInfo, settings,
                                           activationInfo);
}

NeonBatchMatMulWorkload::NeonBatchMatMulWorkload(const BatchMatMulQueueDescriptor& descriptor,
                                                 const WorkloadInfo& info,
                                                 const bool isFastMathEnabled)
    : NeonBaseWorkload<BatchMatMulQueueDescriptor>(descriptor, info)
{
    if (descriptor.m_Parameters.m_AdjointX || descriptor.m_Parameters.m_AdjointY )
    {
        throw Exception("Support for adjoint not implemented.");
    }
    if (descriptor.m_Parameters.m_DataLayoutX != armnn::DataLayout::NCHW
     || descriptor.m_Parameters.m_DataLayoutY != armnn::DataLayout::NCHW )
    {
        throw Exception("Only supported the MatMul in the last 2 dimensions");
    }

    m_Data.ValidateInputsOutputs("NeonBatchMatMulWorkload", 2, 1);

    arm_compute::ITensor& inputX = PolymorphicDowncast<IAclTensorHandle*>(m_Data.m_Inputs[0])->GetTensor();
    arm_compute::ITensor& inputY = PolymorphicDowncast<IAclTensorHandle*>(m_Data.m_Inputs[1])->GetTensor();
    arm_compute::ITensor& output = PolymorphicDowncast<IAclTensorHandle*>(m_Data.m_Outputs[0])->GetTensor();

    // GeMM dispatches kernel handles dynamic inputs differently to static so this flag needs to be set
    inputX.info()->set_are_values_constant(false);
    inputY.info()->set_are_values_constant(false);

    const arm_compute::ActivationLayerInfo activationInfo = ConvertAdditionalInfoToAclActivationLayerInfo(descriptor);

    arm_compute::MatMulInfo matMulInfo;
    matMulInfo.adj_lhs(descriptor.m_Parameters.m_TransposeX);
    matMulInfo.adj_rhs(descriptor.m_Parameters.m_TransposeY);

    arm_compute::CpuMatMulSettings settings;
    settings.fast_math(isFastMathEnabled);

    m_MatMulLayer.configure(&inputX, &inputY, &output, matMulInfo, settings, activationInfo);

    // Report Profiling Details
    WorkloadInfo detailsInfo;
    detailsInfo.m_InputTensorInfos = info.m_InputTensorInfos;
    detailsInfo.m_OutputTensorInfos = info.m_OutputTensorInfos;
    ARMNN_REPORT_PROFILING_WORKLOAD_DESC("NeonBatchMatMulWorkload_Construct",
                                         descriptor.m_Parameters,
                                         detailsInfo,
                                         GetGuid());
}

void NeonBatchMatMulWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_NEON_GUID("NeonBatchMatMulWorkload_Execute", this->GetGuid());
    m_MatMulLayer.run();
}
} //namespace armnn
