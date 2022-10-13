//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "NeonBatchMatMulWorkload.hpp"

#include "NeonWorkloadUtils.hpp"

#include <armnn/utility/PolymorphicDowncast.hpp>

#include <armnnUtils/Permute.hpp>

#include <backendsCommon/WorkloadUtils.hpp>

#include <arm_compute/runtime/NEON/functions/NEGEMM.h>

#include <arm_compute/runtime/NEON/functions/NEPermute.h>


namespace armnn
{
arm_compute::Status NeonBatchMatMulValidate(const TensorInfo& inputX,
                                            const TensorInfo& inputY,
                                            const TensorInfo& output,
                                            const BatchMatMulDescriptor& descriptor)
{
    if (descriptor.m_AdjointX || descriptor.m_AdjointY )
    {
        throw Exception("Support for adjoint not implemented.");
    }
    if (descriptor.m_DataLayoutX != armnn::DataLayout::NCHW || descriptor.m_DataLayoutY != armnn::DataLayout::NCHW )
    {
        throw Exception("Only supported the MatMul in the last 2 dimensions");
    }

    const auto aclInputXInfo = armcomputetensorutils::BuildArmComputeTensorInfo(inputX, descriptor.m_DataLayoutX);
    const auto aclInputYInfo = armcomputetensorutils::BuildArmComputeTensorInfo(inputY, descriptor.m_DataLayoutY);
    const auto aclOutputInfo = armcomputetensorutils::BuildArmComputeTensorInfo(output);

    arm_compute::Status statusGEMM = arm_compute::Status(arm_compute::ErrorCode::OK);
    arm_compute::Status statusPermuteX = arm_compute::Status(arm_compute::ErrorCode::OK);
    arm_compute::Status statusPermuteY = arm_compute::Status(arm_compute::ErrorCode::OK);

    arm_compute::TensorInfo aclPermutedXInfo = arm_compute::TensorInfo();
    arm_compute::TensorInfo aclPermutedYInfo = arm_compute::TensorInfo();

    if (descriptor.m_TransposeX == true)
    {
        auto permutationXVector = GeneratePermutationVectorOnLastTwoDimensions(inputX.GetNumDimensions());
        const auto aclPermutationXVector = armcomputetensorutils::BuildArmComputePermutationVector(permutationXVector);
        const TensorInfo permutedXInfo = armnnUtils::Permuted(inputX, permutationXVector);
        aclPermutedXInfo = armcomputetensorutils::BuildArmComputeTensorInfo(permutedXInfo);

        statusPermuteX = arm_compute::NEPermute::validate(&aclInputXInfo,
                                                          &aclPermutedXInfo,
                                                          aclPermutationXVector);
    }

    if (descriptor.m_TransposeY == true)
    {
        auto permutationYVector = GeneratePermutationVectorOnLastTwoDimensions(inputY.GetNumDimensions());
        const auto aclPermutationYVector = armcomputetensorutils::BuildArmComputePermutationVector(permutationYVector);
        const TensorInfo permutedYInfo = armnnUtils::Permuted(inputY, permutationYVector);
        aclPermutedYInfo = armcomputetensorutils::BuildArmComputeTensorInfo(permutedYInfo);

        statusPermuteY = arm_compute::NEPermute::validate(&aclInputYInfo,
                                                          &aclPermutedYInfo,
                                                          aclPermutationYVector);
    }

    const arm_compute::GEMMInfo& gemm_info = arm_compute::GEMMInfo(false,  // is inputX reshaped
                                                                   false,  // is inputY reshaped
                                                                   false); // is inputY reshaped only 1st run

    statusGEMM = arm_compute::NEGEMM::validate(descriptor.m_TransposeX ? &aclPermutedXInfo : &aclInputXInfo,
                                               descriptor.m_TransposeY ? &aclPermutedYInfo : &aclInputYInfo,
                                               nullptr,
                                               &aclOutputInfo,
                                               1.0,
                                               0,
                                               gemm_info);

    if (statusPermuteX.error_code() == arm_compute::ErrorCode::OK &&
        statusPermuteY.error_code() == arm_compute::ErrorCode::OK &&
        statusGEMM.error_code()     == arm_compute::ErrorCode::OK)
    {
        return arm_compute::Status(arm_compute::ErrorCode::OK,
                                   "All BatchMatMul layers validate status OK.");
    }
    else
    {
        return arm_compute::Status(arm_compute::ErrorCode::RUNTIME_ERROR,
                                   "BatchMatMul layer validate status failed."
                                   + statusGEMM.error_description()
                                   + statusPermuteX.error_description()
                                   + statusPermuteY.error_description());
    }

}

NeonBatchMatMulWorkload::NeonBatchMatMulWorkload(
    const BatchMatMulQueueDescriptor& descriptor, const WorkloadInfo& info)
    : NeonBaseWorkload<BatchMatMulQueueDescriptor>(descriptor, info)
{
    if (descriptor.m_Parameters.m_AdjointX || descriptor.m_Parameters.m_AdjointY )
    {
        throw Exception("Support for adjoint not implemented.");
    }
    if (descriptor.m_Parameters.m_DataLayoutX != armnn::DataLayout::NCHW ||
        descriptor.m_Parameters.m_DataLayoutY != armnn::DataLayout::NCHW )
    {
        throw Exception("Only supported the MatMul in the last 2 dimensions");
    }

    // Report Profiling Details
    ARMNN_REPORT_PROFILING_WORKLOAD_DESC("NeonBatchMatMulWorkload_Construct",
                                         descriptor.m_Parameters,
                                         info,
                                         this->GetGuid());

    m_Data.ValidateInputsOutputs("NeonBatchMatMulWorkload", 2, 1);

    arm_compute::ITensor& inputX = PolymorphicDowncast<IAclTensorHandle*>(m_Data.m_Inputs[0])->GetTensor();
    arm_compute::ITensor& inputY = PolymorphicDowncast<IAclTensorHandle*>(m_Data.m_Inputs[1])->GetTensor();
    auto outputHandle = PolymorphicDowncast<IAclTensorHandle*>(m_Data.m_Outputs[0]);
    arm_compute::ITensor& output = outputHandle->GetTensor();

    arm_compute::DataLayout aclDataLayoutX = ConvertDataLayout(m_Data.m_Parameters.m_DataLayoutX);
    arm_compute::DataLayout aclDataLayoutY = ConvertDataLayout(m_Data.m_Parameters.m_DataLayoutY);

    inputX.info()->set_data_layout(aclDataLayoutX);
    inputY.info()->set_data_layout(aclDataLayoutY);

    if (descriptor.m_Parameters.m_TransposeX == true)
    {
        armnn::PermutationVector permutationXVector
                = GeneratePermutationVectorOnLastTwoDimensions(info.m_InputTensorInfos[0].GetNumDimensions());
        const TensorInfo permutedXInfo = armnnUtils::Permuted(info.m_InputTensorInfos[0], permutationXVector);
        const auto aclPermutationXVector = armcomputetensorutils::BuildArmComputePermutationVector(permutationXVector);

        auto permuteLayerX = std::make_unique<arm_compute::NEPermute>();
        BuildArmComputeTensor(m_PermutedTensorX, permutedXInfo);
        InitialiseArmComputeTensorEmpty(m_PermutedTensorX);
        permuteLayerX->configure(&inputX, &m_PermutedTensorX, aclPermutationXVector);
        m_PermuteLayerX.reset(permuteLayerX.release());
    }

    if (descriptor.m_Parameters.m_TransposeY == true)
    {
        armnn::PermutationVector permutationYVector
                = GeneratePermutationVectorOnLastTwoDimensions(info.m_InputTensorInfos[1].GetNumDimensions());
        const TensorInfo permutedYInfo = armnnUtils::Permuted(info.m_InputTensorInfos[1], permutationYVector);
        const auto aclPermutationYVector = armcomputetensorutils::BuildArmComputePermutationVector(permutationYVector);

        auto permuteLayerY = std::make_unique<arm_compute::NEPermute>();
        BuildArmComputeTensor(m_PermutedTensorY, permutedYInfo);
        InitialiseArmComputeTensorEmpty(m_PermutedTensorY);
        permuteLayerY->configure(&inputY, &m_PermutedTensorY, aclPermutationYVector);
        m_PermuteLayerY.reset(permuteLayerY.release());
    }

    const arm_compute::GEMMInfo& gemm_info = arm_compute::GEMMInfo(false,  // is inputX reshaped
                                                                   false,  // is inputY reshaped
                                                                   false); // is inputY reshaped only 1st run
    auto gemmLayer = std::make_unique<arm_compute::NEGEMM>();
    gemmLayer->configure(descriptor.m_Parameters.m_TransposeX ? &m_PermutedTensorX : &inputX,
                         descriptor.m_Parameters.m_TransposeY ? &m_PermutedTensorY : &inputY,
                         nullptr,
                         &output,
                         1.0,
                         0,
                         gemm_info);
    m_GEMMLayer.reset(gemmLayer.release());
}

void NeonBatchMatMulWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_NEON_GUID("NeonBatchMatMulWorkload_Execute", this->GetGuid());
    if (m_PermuteLayerX)
    {
        m_PermuteLayerX->run();
    }
    if (m_PermuteLayerY)
    {
        m_PermuteLayerY->run();
    }
    m_GEMMLayer->run();
}
} //namespace armnn
