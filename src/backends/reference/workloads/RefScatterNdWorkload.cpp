//
// Copyright © 2024 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <fmt/format.h>
#include "RefScatterNdWorkload.hpp"
#include "RefWorkloadUtils.hpp"
#include "ScatterNd.hpp"
#include "Profiling.hpp"

namespace armnn
{

    RefScatterNdWorkload::RefScatterNdWorkload(const ScatterNdQueueDescriptor& descriptor, const WorkloadInfo& info)
        : RefBaseWorkload(descriptor, info)
    {}

    void RefScatterNdWorkload::Execute() const
    {
        Execute(m_Data.m_Inputs, m_Data.m_Outputs);
    }

    void RefScatterNdWorkload::Execute(std::vector<ITensorHandle*> inputs, std::vector<ITensorHandle*> outputs) const
    {
        ARMNN_SCOPED_PROFILING_EVENT_REF_NAME_GUID("RefScatterNdWorkload_Execute");

        if (m_Data.m_Parameters.m_InputEnabled)
        {
            // Getting TensorInfos for three inputs slots
            const TensorInfo& inputInfo = GetTensorInfo(inputs[0]);
            const TensorInfo& indicesInfo = GetTensorInfo(inputs[1]);
            const TensorInfo& updatesInfo = GetTensorInfo(inputs[2]);

            // Getting Decoder for input
            std::unique_ptr<Decoder<float>> inputDecoder = MakeDecoder<float>(GetTensorInfo(inputs[0]),
                                                                              inputs[0]->Map());

            // Getting Decoder for indices
            std::unique_ptr<Decoder<int>> indicesDecoder = MakeDecoder<int>(GetTensorInfo(inputs[1]),
                                                                            inputs[1]->Map());

            // Getting Decoder for updates
            std::unique_ptr<Decoder<float>> updatesDecoder = MakeDecoder<float>(GetTensorInfo(inputs[2]),
                                                                                inputs[2]->Map());

            // Getting Encoder for output
            std::unique_ptr<Encoder<float>> outputEncoder = MakeEncoder<float>(GetTensorInfo(outputs[0]),
                                                                               outputs[0]->Map());

            ScatterNd(inputInfo,
                      indicesInfo,
                      updatesInfo,
                      *inputDecoder,
                      *indicesDecoder,
                      *updatesDecoder,
                      *outputEncoder,
                      m_Data.m_Parameters);
        }
        else
        {
            // Getting TensorInfos for three inputs slots
            const TensorInfo& shapeInfo = GetTensorInfo(inputs[0]);
            const TensorInfo& indicesInfo = GetTensorInfo(inputs[1]);
            const TensorInfo& updatesInfo = GetTensorInfo(inputs[2]);

            // Getting Decoder for shape
            std::unique_ptr<Decoder<int>> shapeDecoder = MakeDecoder<int>(GetTensorInfo(inputs[0]),
                                                                          inputs[0]->Map());

            // Getting Decoder for indices
            std::unique_ptr<Decoder<int>> indicesDecoder = MakeDecoder<int>(GetTensorInfo(inputs[1]),
                                                                            inputs[1]->Map());

            // Getting Decoder for updates
            std::unique_ptr<Decoder<float>> updatesDecoder = MakeDecoder<float>(GetTensorInfo(inputs[2]),
                                                                                inputs[2]->Map());

            // Getting Encoder for output
            std::unique_ptr<Encoder<float>> outputEncoder = MakeEncoder<float>(GetTensorInfo(outputs[0]),
                                                                               outputs[0]->Map());

            ScatterNd(indicesInfo,
                      updatesInfo,
                      shapeInfo,
                      *indicesDecoder,
                      *updatesDecoder,
                      *shapeDecoder,
                      *outputEncoder,
                      m_Data.m_Parameters);
        }
    }

} // namespace armnn