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
    BatchMatMul(const BatchMatMulDescriptor& params,
                const TensorInfo& inputXInfo,
                const TensorInfo& inputYInfo,
                const TensorInfo& outputInfo,
                Decoder<float>& inputXDecoder,
                Decoder<float>& inputYDecoder,
                Encoder<float>& outputEncoder);

private:
    enum DataSlot
    {
        InputX = 0,
        InputY = 1,
        Output = 2
    };

    const BatchMatMulDescriptor& params;
    TensorInfo inputXInfo;
    TensorInfo inputYInfo;
    TensorInfo outputInfo;
    Decoder<float>& inputXDecoder;
    Decoder<float>& inputYDecoder;
    Encoder<float>& outputEncoder;

    std::vector<float> inputXData;
    std::vector<float> inputYData;

    void ApplyBatchMatMul();

    void ApplyParams();

    void Transpose(DataSlot type);

    void Adjoint(DataSlot type);

    void RecurseTensor(const TensorInfo& tensorInfo,
                       std::function<void(const std::vector<unsigned int>&)> const& operation,
                       std::vector<unsigned int>& curIdx,
                       unsigned int curDim);

    // Adjusts it for when input tensors are of unequal rank
    void AdjustAxesToMulForUnequalRanks(std::pair<unsigned int, unsigned int>& axesXToMul,
                                        std::pair<unsigned int, unsigned int>& axesYToMul);

    float GetValueAt(DataSlot type, std::vector<unsigned int> idx, const std::vector<float>& customData = {});

    void SetValueAt(float value, DataSlot type, std::vector<unsigned int> idx);

    // Takes into account broadcasting
    void AdjustToSafeIdx(DataSlot type, std::vector<unsigned int>& idx);

    unsigned int CalcFlatIdx(DataSlot type, const std::vector<unsigned int>& idx);
};

} // namespace armnn