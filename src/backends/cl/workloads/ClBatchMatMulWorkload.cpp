//
// Copyright © 2022-2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ClBatchMatMulWorkload.hpp"

#include "ClWorkloadUtils.hpp"

#include <aclCommon/ArmComputeTensorUtils.hpp>
#include <aclCommon/ArmComputeUtils.hpp>

#include <armnn/utility/PolymorphicDowncast.hpp>

#include <armnnUtils/Permute.hpp>

#include <backendsCommon/WorkloadUtils.hpp>

#include <cl/ClTensorHandle.hpp>

#include <arm_compute/runtime/CL/functions/CLGEMM.h>
#include <arm_compute/runtime/CL/functions/CLPermute.h>


namespace armnn
{
arm_compute::Status ClBatchMatMulValidate(const TensorInfo& inputX,
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

    arm_compute::Status statusGEMM = arm_compute::Status(arm_compute::ErrorCode::OK);
    arm_compute::Status statusPermuteX = arm_compute::Status(arm_compute::ErrorCode::OK);
    arm_compute::Status statusPermuteY = arm_compute::Status(arm_compute::ErrorCode::OK);

    const auto aclInputXInfo = armcomputetensorutils::BuildArmComputeTensorInfo(inputX, descriptor.m_DataLayoutX);
    const auto aclInputYInfo = armcomputetensorutils::BuildArmComputeTensorInfo(inputY, descriptor.m_DataLayoutY);
    const auto aclOutputInfo = armcomputetensorutils::BuildArmComputeTensorInfo(output);

    arm_compute::TensorInfo aclPermutedXInfo = arm_compute::TensorInfo();
    arm_compute::TensorInfo aclPermutedYInfo = arm_compute::TensorInfo();

    if (descriptor.m_TransposeX == true)
    {
        auto permutationXVector = GeneratePermutationVectorOnLastTwoDimensions(inputX.GetNumDimensions());
        const auto aclPermutationXVector = armcomputetensorutils::BuildArmComputePermutationVector(permutationXVector);
        const TensorInfo permutedXInfo = armnnUtils::Permuted(inputX, permutationXVector);
        aclPermutedXInfo = armcomputetensorutils::BuildArmComputeTensorInfo(permutedXInfo);

        statusPermuteX =  arm_compute::CLPermute::validate(&aclInputXInfo,
                                                           &aclPermutedXInfo,
                                                           aclPermutationXVector);
    }

    if ( descriptor.m_TransposeY == true)
    {
        auto permutationYVector = GeneratePermutationVectorOnLastTwoDimensions(inputY.GetNumDimensions());
        const auto aclPermutationYVector = armcomputetensorutils::BuildArmComputePermutationVector(permutationYVector);
        const TensorInfo permutedYInfo = armnnUtils::Permuted(inputY, permutationYVector);
        aclPermutedYInfo = armcomputetensorutils::BuildArmComputeTensorInfo(permutedYInfo);

        statusPermuteY =  arm_compute::CLPermute::validate(&aclInputYInfo,
                                                           &aclPermutedYInfo,
                                                           aclPermutationYVector);

    }

    const arm_compute::GEMMInfo& gemm_info = arm_compute::GEMMInfo(false,  // is inputX reshaped
                                                                   false,  // is inputY reshaped
                                                                   false); // is inputY reshaped only 1st run


    statusGEMM = arm_compute::CLGEMM::validate(descriptor.m_TransposeX ? &aclPermutedXInfo : &aclInputXInfo,
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
                                   "All Batch Mat Mul layers validate status OK.");
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

    const arm_compute::ICLTensor& inputX = PolymorphicDowncast<ClTensorHandle*>(m_Data.m_Inputs[0])->GetTensor();
    const arm_compute::ICLTensor& inputY = PolymorphicDowncast<ClTensorHandle*>(m_Data.m_Inputs[1])->GetTensor();
    arm_compute::ICLTensor& output = PolymorphicDowncast<ClTensorHandle*>(m_Data.m_Outputs[0])->GetTensor();

    inputX.info()->set_data_layout(armcomputetensorutils::ConvertDataLayout(m_Data.m_Parameters.m_DataLayoutX));
    inputY.info()->set_data_layout(armcomputetensorutils::ConvertDataLayout(m_Data.m_Parameters.m_DataLayoutY));

    arm_compute::TensorInfo aclPermutedXInfo = arm_compute::TensorInfo();
    arm_compute::TensorInfo aclPermutedYInfo = arm_compute::TensorInfo();

    if (descriptor.m_Parameters.m_TransposeX == true)
    {
        armnn::PermutationVector permutationXVector
                = GeneratePermutationVectorOnLastTwoDimensions(info.m_InputTensorInfos[0].GetNumDimensions());
        const TensorInfo permutedXInfo = armnnUtils::Permuted(info.m_InputTensorInfos[0], permutationXVector);
        const auto aclPermutationXVector = armcomputetensorutils::BuildArmComputePermutationVector(permutationXVector);
        armcomputetensorutils::BuildArmComputeTensor(m_PermutedTensorX, permutedXInfo);
        armcomputetensorutils::InitialiseArmComputeTensorEmpty(m_PermutedTensorX);

        auto permuteLayerX = std::make_unique<arm_compute::CLPermute>();
        permuteLayerX->configure(clCompileContext,
                                 &inputX,
                                 &m_PermutedTensorX,
                                 aclPermutationXVector);
        m_PermuteLayerX.reset(permuteLayerX.release());
    }

    if (descriptor.m_Parameters.m_TransposeY == true)
    {
        armnn::PermutationVector permutationYVector
                = GeneratePermutationVectorOnLastTwoDimensions(info.m_InputTensorInfos[1].GetNumDimensions());
        const TensorInfo permutedYInfo = armnnUtils::Permuted(info.m_InputTensorInfos[1], permutationYVector);
        const auto aclPermutationYVector = armcomputetensorutils::BuildArmComputePermutationVector(permutationYVector);
        armcomputetensorutils::BuildArmComputeTensor(m_PermutedTensorY, permutedYInfo);
        armcomputetensorutils::InitialiseArmComputeTensorEmpty(m_PermutedTensorY);

        auto permuteLayerY = std::make_unique<arm_compute::CLPermute>();
        permuteLayerY->configure(clCompileContext,
                                 &inputY,
                                 &m_PermutedTensorY,
                                 aclPermutationYVector);
        m_PermuteLayerY.reset(permuteLayerY.release());
    }

    const arm_compute::GEMMInfo& gemm_info = arm_compute::GEMMInfo(false,  // is inputX reshaped
                                                                   false,  // is inputY reshaped
                                                                   false); // is inputY reshaped only 1st run
    auto gemmLayer = std::make_unique<arm_compute::CLGEMM>();
    gemmLayer->configure(clCompileContext,
                         descriptor.m_Parameters.m_TransposeX ? &m_PermutedTensorX : &inputX,
                         descriptor.m_Parameters.m_TransposeY ? &m_PermutedTensorY : &inputY,
                         nullptr,
                         &output,
                         1.0,
                         0,
                         gemm_info);
    m_GEMMLayer.reset(gemmLayer.release());
}

void ClBatchMatMulWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_CL_GUID("ClBatchMatMulWorkload_Execute", this->GetGuid());
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
