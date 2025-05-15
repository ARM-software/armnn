//
// Copyright © 2025 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "TosaTestUtils.hpp"

using namespace armnn;
using namespace tosa;

void VerifySoftmax(TosaSerializationBasicBlock* softmaxBlock,
                   std::vector<std::vector<int32_t>> inputShape,
                   std::vector<std::vector<int32_t>> outputShape,
                   const BaseDescriptor& softmaxDescriptor)
{
    uint32_t numInputs = static_cast<uint32_t>(inputShape.size());
    uint32_t numOutputs = static_cast<uint32_t>(outputShape.size());

    CHECK(numInputs == 1);
    CHECK(numOutputs == 1);

    std::string blockStr = "Op_SOFTMAX_block_";
    CHECK(softmaxBlock->GetName().find(blockStr)  != std::string::npos);
    CHECK(softmaxBlock->GetInputs().size() == 1);
    CHECK(softmaxBlock->GetOutputs().size() == 1);
    CHECK(softmaxBlock->GetOperators().size() == 58);
    CHECK(softmaxBlock->GetTensors().size() == 59);

    TosaSerializationOperator* rescaleOp = softmaxBlock->GetOperators().at(0);

    VerifyTosaAttribute(softmaxDescriptor,
                        rescaleOp->GetAttribute(),
                        inputShape[0],
                        outputShape[0],
                        LayerType::Softmax);
}