//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "RefFakeQuantizationFloat32Workload.hpp"

#include "RefWorkloadUtils.hpp"

#include "Profiling.hpp"

#include <armnn/utility/NumericCast.hpp>

namespace armnn
{

void FakeQuantization(const float* inputData, float* outputData, uint32_t numElements, float min, float max)
{
    float scale = (max - min) / 255.f;
    int32_t offset = armnn::numeric_cast<int32_t>((-min * 255.f) / (max - min));

    for (uint32_t i = 0; i < numElements; i++)
    {
        outputData[i] = static_cast<float>(armnn::Quantize<uint8_t>(inputData[i], scale, offset));
    }

}

void RefFakeQuantizationFloat32Workload::Execute() const
{
    Execute(m_Data.m_Inputs, m_Data.m_Outputs);
}

void RefFakeQuantizationFloat32Workload::ExecuteAsync(WorkingMemDescriptor &workingMemDescriptor)
{
    Execute(workingMemDescriptor.m_Inputs, workingMemDescriptor.m_Outputs);
}

void RefFakeQuantizationFloat32Workload::Execute(std::vector<ITensorHandle*> inputs,
                                                 std::vector<ITensorHandle*> outputs) const
{
    ARMNN_SCOPED_PROFILING_EVENT(Compute::CpuRef, "RefFakeQuantizationFloat32Workload_Execute");

    const TensorInfo& inputInfo = GetTensorInfo(inputs[0]);

    const float* inputData = reinterpret_cast<const float*>(inputs[0]->Map());
    float* outputData = reinterpret_cast<float*>(outputs[0]->Map());
    FakeQuantization(inputData, outputData, inputInfo.GetNumElements(),
                     m_Data.m_Parameters.m_Min,
                     m_Data.m_Parameters.m_Max);
}

} //namespace armnn
