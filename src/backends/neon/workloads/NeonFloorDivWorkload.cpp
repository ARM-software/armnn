//
// Copyright © 2024 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "NeonFloorDivWorkload.hpp"
#include "NeonWorkloadUtils.hpp"
#include <armnn/utility/PolymorphicDowncast.hpp>
#include <aclCommon/ArmComputeUtils.hpp>
#include "backendsCommon/WorkloadUtils.hpp"

namespace armnn
{

/// Utility function used for the two cast layer inputs to convert the output layer tensor types.
inline TensorInfo ConvertTensorToFloat32(const TensorInfo& tensorInfo)
{
    // Change datatype of tensor info and return the new tensor
    TensorInfo newTensorInfo(tensorInfo);
    newTensorInfo.SetDataType(DataType::Float32);
    return newTensorInfo;
}

/// Utility function used to check if a vector of tensors are Signed32
inline bool IsAllTensorsSigned32(const std::vector<TensorInfo>& tensorInfos)
{
    for (const auto& tensorInfo : tensorInfos)
    {
        // For every tensorInfo, check the data type, return false if not Signed32
        if(tensorInfo.GetDataType() != armnn::DataType::Signed32)
        {
            return false;
        }
    }
    return true;
}

/// Utility function used to check if statuses are returning 'OK'
inline bool IsValidationPassing(const std::vector<arm_compute::Status>& statuses)
{
    // For each status, check if code is 'OK', if not, return false
    for (const auto& status : statuses)
    {
        if(status.error_code() != arm_compute::ErrorCode::OK)
        {
            return false;
        }
    }
    return true;
}

arm_compute::Status NeonFloorDivWorkloadValidate(const TensorInfo& input0Info,
                                                 const TensorInfo& input1Info,
                                                 const TensorInfo& outputInfo,
                                                 const ActivationDescriptor* activationDescriptor)
{
    // Transform ArmNN TensorInfo to ACL TensorInfo
    const arm_compute::TensorInfo inputInfo0 = BuildArmComputeTensorInfo(input0Info);
    const arm_compute::TensorInfo inputInfo1 = BuildArmComputeTensorInfo(input1Info);
    const arm_compute::TensorInfo outputInfo0 = BuildArmComputeTensorInfo(outputInfo);

    const arm_compute::ActivationLayerInfo activationInfo = ConvertActivationDescriptorToAclActivationLayerInfo(
        activationDescriptor);

    // If Tensors are Signed32 we need to Cast them to floats, this is to ensure we get the correct
    // output if the result is a negative number, as we should floor towards -(infinity)
    if (IsAllTensorsSigned32({input0Info, input1Info, outputInfo}))
    {
        // Validate Cast0
        TensorInfo outputCast0_Info = ConvertTensorToFloat32(input0Info);
        const arm_compute::TensorInfo outputCast0 = BuildArmComputeTensorInfo(outputCast0_Info);

        auto statusCast0 = arm_compute::NECast::validate(&inputInfo0,
                                                         &outputCast0,
                                                         arm_compute::ConvertPolicy::WRAP);
        // Validate Cast1
        TensorInfo outputCast1_Info = ConvertTensorToFloat32(input1Info);
        const arm_compute::TensorInfo outputCast1 = BuildArmComputeTensorInfo(outputCast1_Info);

        auto statusCast1 = arm_compute::NECast::validate(&inputInfo1,
                                                         &outputCast1,
                                                         arm_compute::ConvertPolicy::WRAP);

        // Validate Div
        TensorInfo outputDiv_Info = ConvertTensorToFloat32(outputInfo);
        const arm_compute::TensorInfo outputDivInfo = BuildArmComputeTensorInfo(outputDiv_Info);

        auto statusDiv = arm_compute::NEElementwiseDivision::validate(&outputCast0,
                                                                      &outputCast1,
                                                                      &outputDivInfo,
                                                                      activationInfo);
        // Validate Floor
        TensorInfo outputFloor_Info = ConvertTensorToFloat32(outputInfo);
        const arm_compute::TensorInfo outputFloorInfo = BuildArmComputeTensorInfo(outputFloor_Info);

        auto statusFloor = arm_compute::NEFloor::validate(&outputDivInfo,
                                                          &outputFloorInfo);
        // Validate Cast2
        auto statusCast2 = arm_compute::NECast::validate(&outputFloorInfo,
                                                         &outputInfo0,
                                                         arm_compute::ConvertPolicy::WRAP);

        // Return OK if all the layers are valid
        if (IsValidationPassing({statusCast0, statusCast1, statusDiv, statusFloor, statusCast2}))
        {
            return arm_compute::Status(arm_compute::ErrorCode::OK);
        }
    }
    else
    {
        // Validate Div
        auto statusDiv = arm_compute::NEElementwiseDivision::validate(&inputInfo0,
                                                                      &inputInfo1,
                                                                      &outputInfo0,
                                                                      activationInfo);
        // Validate Floor
        auto statusFloor = arm_compute::NEFloor::validate(&outputInfo0,
                                                          &outputInfo0);
        // Return OK if all the layers are valid
        if (IsValidationPassing({statusDiv, statusFloor}))
        {
            return arm_compute::Status(arm_compute::ErrorCode::OK);
        }
    }
    return arm_compute::Status(arm_compute::ErrorCode::RUNTIME_ERROR,
                               "NeonFloorDivValidation: FloorDiv layer validation failed.");
}

NeonFloorDivWorkload::NeonFloorDivWorkload(const DivisionQueueDescriptor& descriptor,
                                           const WorkloadInfo& info)
        : NeonBaseWorkload<DivisionQueueDescriptor>(descriptor, info)
{
    m_Data.ValidateInputsOutputs("NeonFloorDivWorkload", 2, 1);

    TensorInfo input0Info = info.m_InputTensorInfos[0];
    TensorInfo input1Info = info.m_InputTensorInfos[1];
    TensorInfo outputInfo = info.m_OutputTensorInfos[0];

    arm_compute::ITensor& input0   = PolymorphicDowncast<IAclTensorHandle*>(m_Data.m_Inputs[0])->GetTensor();
    arm_compute::ITensor& input1 = PolymorphicDowncast<IAclTensorHandle*>(m_Data.m_Inputs[1])->GetTensor();
    arm_compute::ITensor& output  = PolymorphicDowncast<IAclTensorHandle*>(m_Data.m_Outputs[0])->GetTensor();

    // Get data type of input and output
    arm_compute::DataType inputDataType  = PolymorphicDowncast<NeonTensorHandle*>(m_Data.m_Inputs[0])->GetDataType();
    arm_compute::DataType outputDataType = PolymorphicDowncast<NeonTensorHandle*>(m_Data.m_Outputs[0])->GetDataType();

    const arm_compute::ActivationLayerInfo activationInfo =
        ConvertAdditionalInfoToAclActivationLayerInfo(descriptor);

    // If Tensors are Signed32 we need to Cast them to floats, this is to ensure we get the correct
    // output if the result is a negative number, as we should floor towards -(infinity)
    if(inputDataType == arm_compute::DataType::S32 && outputDataType == arm_compute::DataType::S32)
    {
        // Create new Cast layer pointers if type is S32
        m_CastLayer0.reset(new arm_compute::NECast());
        m_CastLayer1.reset(new arm_compute::NECast());
        m_CastLayer2.reset(new arm_compute::NECast());

        // Cast Input 0 to type float32
        TensorInfo outputCast0_Info = ConvertTensorToFloat32(input0Info);

        // Initialise output tensor based on Float32 type
        BuildArmComputeTensor(m_OutputCast0, outputCast0_Info);
        armcomputetensorutils::InitialiseArmComputeTensorEmpty(m_OutputCast0);

        // Configure first Cast Layer
        m_CastLayer0->configure(&input0, &m_OutputCast0, arm_compute::ConvertPolicy::WRAP);

        // Cast Input 1 to type Float32
        TensorInfo outputCast1_Info = ConvertTensorToFloat32(input1Info);

        // Initialise Output tensor based on Float32 type
        BuildArmComputeTensor(m_OutputCast1, outputCast1_Info);
        armcomputetensorutils::InitialiseArmComputeTensorEmpty(m_OutputCast1);

        // Configure second Cast Layer
        m_CastLayer1->configure(&input1, &m_OutputCast1, arm_compute::ConvertPolicy::WRAP);

        // Create Div output tensor
        TensorInfo outputDiv_Info = ConvertTensorToFloat32(outputInfo);
        BuildArmComputeTensor(m_OutputDiv, outputDiv_Info);
        armcomputetensorutils::InitialiseArmComputeTensorEmpty(m_OutputDiv);

        // Configure Div Layer
        m_DivLayer.configure(&m_OutputCast0, &m_OutputCast1, &m_OutputDiv, activationInfo);

        // Create Floor output tensor
        BuildArmComputeTensor(m_OutputFloor, outputDiv_Info);
        armcomputetensorutils::InitialiseArmComputeTensorEmpty(m_OutputFloor);

        // Configure Floor Layer
        m_FloorLayer.configure(&m_OutputDiv, &m_OutputFloor);

        // Configure third Cast Layer
        m_CastLayer2->configure(&m_OutputFloor, &output, arm_compute::ConvertPolicy::WRAP);
    }
    else
    {
        // Create Div output tensor
        BuildArmComputeTensor(m_OutputDiv, outputInfo);
        armcomputetensorutils::InitialiseArmComputeTensorEmpty(m_OutputDiv);

        // Configure Div Layer
        m_DivLayer.configure(&input0, &input1, &m_OutputDiv, activationInfo);

        // Configure Floor Layer
        m_FloorLayer.configure(&m_OutputDiv, &output);
    }
}

void NeonFloorDivWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_NEON_NAME_GUID("NeonFloorDivWorkload_Execute");
    // Only run Cast Layers if needed. e.g. if it exists
    if(m_CastLayer0 && m_CastLayer1)
    {
        m_CastLayer0->run();
        m_CastLayer1->run();

        // Delete objects after running
        m_CastLayer0.reset();
        m_CastLayer1.reset();
    }
    m_DivLayer.run();
    m_FloorLayer.run();
    if(m_CastLayer2)
    {
        m_CastLayer2->run();

        // Delete object after running layer
        m_CastLayer2.reset();
    }
}
} //namespace armnn