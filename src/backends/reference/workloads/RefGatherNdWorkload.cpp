//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "RefGatherNdWorkload.hpp"

#include "Gather.hpp"
#include "Profiling.hpp"
#include "RefWorkloadUtils.hpp"
#include "backendsCommon/WorkloadUtils.hpp"

namespace armnn
{

void RefGatherNdWorkload::Execute() const
{
    Execute(m_Data.m_Inputs, m_Data.m_Outputs);
}

void RefGatherNdWorkload::ExecuteAsync(ExecutionData& executionData)
{
    WorkingMemDescriptor* workingMemDescriptor = static_cast<WorkingMemDescriptor*>(executionData.m_Data);
    Execute(workingMemDescriptor->m_Inputs, workingMemDescriptor->m_Outputs);
}

void RefGatherNdWorkload::Execute(std::vector<ITensorHandle*> inputs, std::vector<ITensorHandle*> outputs) const
{
    ARMNN_SCOPED_PROFILING_EVENT(Compute::CpuRef, "RefGatherNdWorkload_Execute");

    const TensorInfo& inputInfo0 = GetTensorInfo(inputs[0]);
    const TensorInfo& inputInfo1 = GetTensorInfo(inputs[1]);
    const TensorInfo& outputInfo = GetTensorInfo(outputs[0]);

    std::unique_ptr<Decoder<float>> params_decoderPtr = MakeDecoder<float>(inputInfo0, inputs[0]->Map());

    const int32_t* indicesDataPtr = reinterpret_cast<int32_t*>(inputs[1]->Map());
    std::vector<int32_t> indices(indicesDataPtr, indicesDataPtr + inputInfo1.GetNumElements());

    std::unique_ptr<Encoder<float>> output_encoderPtr = MakeEncoder<float>(outputInfo, outputs[0]->Map());

    std::map<std::string, unsigned int> keyIndices = CalculateGatherNdKeyIndices(inputInfo0, inputInfo1);

    /// Calculate flattened indices: flattenedIndices = indices * flattenedCoefficients
    // Calculate the flattened coefficients to use in the multiplication
    // to calculate the flattened indices needed by gather
    TensorShape paramsShape = inputInfo0.GetShape();
    std::vector<unsigned int> flattenedCoeff(keyIndices["ND"], 1);
    for (unsigned int i = 1; i < keyIndices["ND"]; ++i)
    {
        flattenedCoeff[i-1] = paramsShape[i];
    }
    for (unsigned int i = keyIndices["ND"]-1; i > 0; --i)
    {
        flattenedCoeff[i-1] *= flattenedCoeff[i];
    }

    // Prepare the vector to store the output of the matrix multiplication,
    // which will represent the flattened indices needed by gather
    armnn::TensorInfo flattenedIndices_Info = inputInfo1;
    flattenedIndices_Info.SetShape({ keyIndices["W"] });
    std::vector<int32_t> flattenedIndices(flattenedIndices_Info.GetNumElements(), 0);

    // Multiplication to calculate the flattened indices, which are the indices needed by gather.
    for (unsigned int i = 0; i < keyIndices["W"]; ++i)
    {
        for (unsigned int j = 0; j < keyIndices["ND"]; ++j)
        {
            flattenedIndices[i] += indices[i * keyIndices["ND"] + j] * static_cast<int32_t>(flattenedCoeff[j]);
        }
    }

    /// Call Gather with adequate shapes
    // Reshape params into {K, C}
    armnn::TensorInfo params_K_C_Info =  inputInfo0;
    params_K_C_Info.SetShape({ keyIndices["K"], keyIndices["C"] });

    // Reshape indices into {N, W}
    armnn::TensorInfo indices_N_W_Info = inputInfo1;
    indices_N_W_Info.SetShape({ keyIndices["N"], keyIndices["W"] });

    // Reshape output to have the shape given by gather {N, W, C}
    // (the original outputInfo has the shape given by gatherNd)
    armnn::TensorInfo outputGather_Info = outputInfo;
    outputGather_Info.SetShape({ keyIndices["N"], keyIndices["W"], keyIndices["C"]  });

    // output_gather = gather(params_K_C, indices_N_W)
    Gather(params_K_C_Info, indices_N_W_Info, outputGather_Info,
           *params_decoderPtr, flattenedIndices.data(), *output_encoderPtr, 0);
}

} //namespace armnn
