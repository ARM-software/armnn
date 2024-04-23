//
// Copyright © 2023-2024 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "TosaTestUtils.hpp"

using namespace armnn;
using namespace tosa;

void VerifySplit(TosaSerializationBasicBlock* splitBlock,
                 std::vector<std::vector<int32_t>> inputShape,
                 std::vector<std::vector<int32_t>> outputShape,
                 const BaseDescriptor& splitDescriptor,
                 DType dataType = DType_FP32)
{
    uint32_t numInputs = static_cast<uint32_t>(inputShape.size());
    uint32_t numOutputs = static_cast<uint32_t>(outputShape.size());

    std::string blockStr = "Op_SPLIT_block_";
    CHECK(splitBlock->GetName().find(blockStr)  != std::string::npos);
    CHECK(splitBlock->GetInputs().size() == numInputs);
    CHECK(splitBlock->GetOutputs().size() == numOutputs);
    CHECK(splitBlock->GetOperators().size() == 3);
    CHECK(splitBlock->GetTensors().size() == 4);

    //
    // Verify slice operator
    //

    for (uint32_t i = 0; i < splitBlock->GetOperators().size(); i++)
    {
        TosaSerializationOperator *sliceOp = splitBlock->GetOperators().at(i);
        uint32_t sliceOpOutputs = 1;
        CHECK(sliceOp->GetInputTensorNames().size() == numInputs);
        CHECK(sliceOp->GetOutputTensorNames().size() == sliceOpOutputs);

        std::basic_string<char> blockInputName = splitBlock->GetInputs()[0];
        std::basic_string<char> operatorInputName = sliceOp->GetInputTensorNames()[0];

        std::string opInputStr = "input_";

        CHECK(blockInputName == operatorInputName);
        CHECK(splitBlock->GetTensorByName(blockInputName));
        CHECK(blockInputName.find(opInputStr) != std::string::npos);

        TosaSerializationTensor* inputTensor = splitBlock->GetTensorByName(operatorInputName);
        CHECK(inputTensor->GetDtype() == dataType);
        CHECK(inputTensor->GetData().size() == 0);
        CHECK(inputTensor->GetShape() == inputShape[0]);

        std::basic_string<char> blockOutputName = splitBlock->GetOutputs()[i];
        std::basic_string<char> operatorOutputName  = sliceOp->GetOutputTensorNames()[0];

        std::string opOutputStr = "output" + std::to_string(i) + "_";

        CHECK(blockOutputName == operatorOutputName);
        CHECK(splitBlock->GetTensorByName(blockOutputName));
        CHECK(blockOutputName.find(opOutputStr)  != std::string::npos);

        TosaSerializationTensor* outputTensor = splitBlock->GetTensorByName(operatorOutputName);
        CHECK(outputTensor->GetDtype() == dataType);
        CHECK(outputTensor->GetData().size() == 0);
        CHECK(outputTensor->GetShape() == outputShape[0]);

        CHECK(sliceOp->GetAttributeType() == Attribute_SliceAttribute);
        CHECK(sliceOp->GetOp() == Op_SLICE);

        VerifyTosaAttribute(splitDescriptor,
                            sliceOp->GetAttribute(),
                            inputShape[0],
                            outputShape[0],
                            LayerType::Splitter,
                            i);
    }

}