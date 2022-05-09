//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "NeonGatherNdWorkload.hpp"
#include "NeonWorkloadUtils.hpp"
#include <armnn/utility/PolymorphicDowncast.hpp>
#include <aclCommon/ArmComputeUtils.hpp>
#include "backendsCommon/WorkloadUtils.hpp"

namespace armnn
{
arm_compute::Status NeonGatherNdWorkloadValidate(const TensorInfo& paramsInfo,
                                                 const TensorInfo& indicesInfo,
                                                 const TensorInfo& outputInfo)
{
    // Calculate ND, K, W, C.
    std::map<std::string, unsigned int> keyIndices = CalculateGatherNdKeyIndices(paramsInfo, indicesInfo);

    /// Validate Mul
    // Indices with shape { W, ND }
    armnn::TensorInfo indices_W_ND_Info = indicesInfo;
    indices_W_ND_Info.SetShape({ keyIndices["W"], keyIndices["ND"] });
    const arm_compute::TensorInfo aclIndicesInfo = BuildArmComputeTensorInfo(indices_W_ND_Info);

    // Flattened coefficients with shape { ND }
    armnn::TensorInfo flattenedCoeff_Info = indicesInfo;
    flattenedCoeff_Info.SetShape({ keyIndices["ND"] });
    const arm_compute::TensorInfo aclFlattenedCoeffInfo = BuildArmComputeTensorInfo(flattenedCoeff_Info);

    // Output of Mul with shape { W, ND }
    const arm_compute::TensorInfo aclOutputMulInfo = BuildArmComputeTensorInfo(indices_W_ND_Info);

    auto statusMul = arm_compute::NEPixelWiseMultiplication::validate(&aclIndicesInfo,
                                                                      &aclFlattenedCoeffInfo,
                                                                      &aclOutputMulInfo,
                                                                      1.0f,
                                                                      arm_compute::ConvertPolicy::WRAP,
                                                                      arm_compute::RoundingPolicy::TO_ZERO,
                                                                      arm_compute::ActivationLayerInfo());

    /// Validate ReduceSum
    // Flattened indices with shape { W }
    armnn::TensorInfo flattenedIndices_Info = indicesInfo;
    flattenedIndices_Info.SetShape({ keyIndices["W"] });
    const arm_compute::TensorInfo aclFlattenedIndicesInfo = BuildArmComputeTensorInfo(flattenedIndices_Info);

    const std::vector<unsigned int> armnnReduceAxes(1, 1);
    arm_compute::Coordinates coords = BuildArmComputeReductionCoordinates(aclOutputMulInfo.num_dimensions(),
                                                                          indices_W_ND_Info.GetNumDimensions(),
                                                                          armnnReduceAxes);

    auto statusReduceSum = arm_compute::NEReductionOperation::validate(&aclOutputMulInfo,
                                                                       &aclFlattenedIndicesInfo,
                                                                       static_cast<unsigned int>(coords[0]),
                                                                       arm_compute::ReductionOperation::SUM,
                                                                       false);

    /// Validate Gather
    // Params with shape { K, C }
    armnn::TensorInfo params_K_C_Info = paramsInfo;
    params_K_C_Info.SetShape({ keyIndices["K"], keyIndices["C"] });
    const arm_compute::TensorInfo aclParamsInfo = BuildArmComputeTensorInfo(params_K_C_Info);

    // Output of gather with shape { W, C }
    armnn::TensorInfo outputGather_Info = outputInfo;
    outputGather_Info.SetShape({ keyIndices["W"], keyIndices["C"] });
    const arm_compute::TensorInfo aclOutputGatherInfo = BuildArmComputeTensorInfo(outputGather_Info);

    auto aclAxis = ComputeAclAxis(0, params_K_C_Info);
    auto statusGather =
            arm_compute::NEGather::validate(&aclParamsInfo, &aclFlattenedIndicesInfo, &aclOutputGatherInfo, aclAxis);

    /// Validate Reshape
    const arm_compute::TensorInfo aclOutputInfo = BuildArmComputeTensorInfo(outputInfo);

    auto statusReshape = arm_compute::NEReshapeLayer::validate(&aclOutputGatherInfo, &aclOutputInfo);

    /// Return OK if all the layers are valid
    auto okCode = arm_compute::ErrorCode::OK;
    if (statusMul.error_code()       == okCode &&
        statusReduceSum.error_code() == okCode &&
        statusGather.error_code()    == okCode &&
        statusReshape.error_code()   == okCode)
    {
        return arm_compute::Status(arm_compute::ErrorCode::OK,
                                   "All GatherND layers validate status OK.");
    }
    else
    {
        return arm_compute::Status(arm_compute::ErrorCode::RUNTIME_ERROR,
                                   "GatherND layer validate status failed.");
    }
}

NeonGatherNdWorkload::NeonGatherNdWorkload(const GatherNdQueueDescriptor& descriptor,
                                           const WorkloadInfo& info)
        : NeonBaseWorkload<GatherNdQueueDescriptor>(descriptor, info)
{
    m_Data.ValidateInputsOutputs("NeonGatherNdWorkload", 2, 1);

    TensorInfo paramsInfo  = info.m_InputTensorInfos[0];
    TensorInfo indicesInfo = info.m_InputTensorInfos[1];
    TensorInfo outputInfo  = info.m_OutputTensorInfos[0];

    arm_compute::ITensor& input   = PolymorphicDowncast<IAclTensorHandle*>(m_Data.m_Inputs[0])->GetTensor();
    arm_compute::ITensor& indices = PolymorphicDowncast<IAclTensorHandle*>(m_Data.m_Inputs[1])->GetTensor();
    arm_compute::ITensor& output  = PolymorphicDowncast<IAclTensorHandle*>(m_Data.m_Outputs[0])->GetTensor();

    // Calculate ND, K, W, C.
    std::map<std::string, unsigned int> keyIndices = CalculateGatherNdKeyIndices(paramsInfo, indicesInfo);

    /// Calculate flattened indices: m_FlattenedIndices = indices * m_FlattenedCoeff.
    /// This could be done using MatMul instead of multiplication followed by reduce sum operation,
    /// but GeMM does not support s32 at the moment.

    // Prepare the tensor to store the output of the reduce_sum operation
    armnn::TensorInfo flattenedIndices_Info = indicesInfo;
    flattenedIndices_Info.SetShape({ keyIndices["W"] });
    BuildArmComputeTensor(m_FlattenedIndices, flattenedIndices_Info);
    armcomputetensorutils::InitialiseArmComputeTensorEmpty(m_FlattenedIndices);

    // Reshape indices into { W, ND }
    indices.info()->set_tensor_shape(BuildArmComputeTensorShape({ keyIndices["W"], keyIndices["ND"] }));

    // Calculate the m_FlattenedCoeff
    TensorShape paramsShape = paramsInfo.GetShape();
    std::vector<int32_t> flattenedCoeff(keyIndices["ND"], 1);
    for (unsigned int i = 1; i < keyIndices["ND"]; ++i)
    {
        flattenedCoeff[i - 1] = static_cast<int32_t>(paramsShape[i]);
    }
    for (unsigned int i = keyIndices["ND"] - 1; i > 0; --i)
    {
        flattenedCoeff[i - 1] *= flattenedCoeff[i];
    }
    armnn::TensorInfo flattenedCoeff_Info = indicesInfo;
    flattenedCoeff_Info.SetShape({ keyIndices["ND"] });
    BuildArmComputeTensor(m_FlattenedCoeff, flattenedCoeff_Info);
    armcomputetensorutils::InitialiseArmComputeTensorEmpty(m_FlattenedCoeff);
    ARMNN_ASSERT_MSG(indicesInfo.GetDataType() == DataType::Signed32,
                     "flattenedCoeff must be same data type as m_FlattenedCoeff");
    CopyArmComputeITensorData<int32_t>(flattenedCoeff.data(), m_FlattenedCoeff);

    // Prepare the tensor to store the output of the multiplication
    armnn::TensorInfo outputMul_Info = indicesInfo;
    outputMul_Info.SetShape({ keyIndices["W"], keyIndices["ND"] });
    BuildArmComputeTensor(m_OutputMul, outputMul_Info);
    armcomputetensorutils::InitialiseArmComputeTensorEmpty(m_OutputMul);

    // Multiply
    m_MulLayer.configure(&indices,
                         &m_FlattenedCoeff,
                         &m_OutputMul,
                         1.0f,
                         arm_compute::ConvertPolicy::WRAP,
                         arm_compute::RoundingPolicy::TO_ZERO,
                         arm_compute::ActivationLayerInfo());

    // Reduce Sum
    const std::vector<unsigned int> armnnReduceAxes(1, 1);
    arm_compute::Coordinates coords = BuildArmComputeReductionCoordinates(m_OutputMul.info()->num_dimensions(),
                                                                          outputMul_Info.GetNumDimensions(),
                                                                          armnnReduceAxes);
    m_ReduceSumLayer.configure(&m_OutputMul,
                               &m_FlattenedIndices,
                               static_cast<unsigned int>(coords[0]),
                               arm_compute::ReductionOperation::SUM,
                               false);

    /// Call Gather with adequate shapes
    // Reshape params into { K, C }
    paramsInfo.SetShape({ keyIndices["K"], keyIndices["C"] });
    input.info()->set_tensor_shape(BuildArmComputeTensorShape(paramsInfo.GetShape()));

    // Reshape output to have the shape given by gather { W, C }
    // (the original outputInfo has the shape given by gatherNd)
    armnn::TensorInfo outputGather_Info = outputInfo;
    outputGather_Info.SetShape({ keyIndices["W"], keyIndices["C"] });
    BuildArmComputeTensor(m_OutputGather, outputGather_Info);
    armcomputetensorutils::InitialiseArmComputeTensorEmpty(m_OutputGather);

    m_GatherLayer.configure(&input, &m_FlattenedIndices, &m_OutputGather, ComputeAclAxis(0, paramsInfo));

    // Reshape output to the original output shape
    m_ReshapeLayer.configure(&m_OutputGather, &output);
}

void NeonGatherNdWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_NEON_GUID("NeonGatherNdWorkload_Execute", this->GetGuid());
    m_MulLayer.run();
    m_ReduceSumLayer.run();
    m_GatherLayer.run();
    m_ReshapeLayer.run();
}
} //namespace armnn