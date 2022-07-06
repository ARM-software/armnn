//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "Encoders.hpp"
#include "Decoders.hpp"

#include <armnn/backends/WorkloadData.hpp>

namespace armnn
{

class BatchMatMul {
public:
    enum DataSlot
    {
        InputX = 0,
        InputY = 1,
        Output = 2
    };

    BatchMatMul(const BatchMatMulDescriptor& params,
                const TensorInfo& inputXInfo,
                const TensorInfo& inputYInfo,
                const TensorInfo& outputInfo,
                Decoder<float>& inputXDecoder,
                Decoder<float>& inputYDecoder,
                Encoder<float>& outputEncoder)
        : params(params),
          inputXInfo(inputXInfo),
          inputYInfo(inputYInfo),
          outputInfo(outputInfo),
          inputXDecoder(inputXDecoder),
          inputYDecoder(inputYDecoder),
          outputEncoder(outputEncoder)
    {}

    void BatchMatMulImpl();

    void RecurseBMM(std::vector<unsigned int>& curIdx, unsigned int curDim);

    // Adjusts it for when input tensors are of unequal rank
    void AdjustAxesToMulForUnequalRanks(
        std::pair<std::pair<unsigned int, unsigned int>, std::pair<unsigned int, unsigned int>>& axesToMul);

    float GetValueAt(DataSlot type, std::vector<unsigned int> idx);

    void SetValueAt(float value, DataSlot type, std::vector<unsigned int> idx);

    // Takes into account broadcasting
    void AdjustToSafeIdx(DataSlot type, std::vector<unsigned int>& idx);

    unsigned int CalcFlatIdx(DataSlot type, const std::vector<unsigned int>& idx);

    template <typename T>
    std::string StringifyVec(const std::vector<T>& vec);

private:
    const BatchMatMulDescriptor& params;
    const TensorInfo& inputXInfo;
    const TensorInfo& inputYInfo;
    const TensorInfo& outputInfo;
    Decoder<float>& inputXDecoder;
    Decoder<float>& inputYDecoder;
    Encoder<float>& outputEncoder;

    std::vector<float> inputXData;
    std::vector<float> inputYData;

};

} // namespace armnn