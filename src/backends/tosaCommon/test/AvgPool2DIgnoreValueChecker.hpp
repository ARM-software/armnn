//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "TosaTestUtils.hpp"

using namespace armnn;
using namespace tosa;

void VerifyAvgPool2DIgnoreValue(TosaSerializationBasicBlock* basicBlock,
                                std::vector<std::vector<int32_t>> inputShape,
                                std::vector<std::vector<int32_t>> outputShape,
                                std::vector<std::vector<int32_t>> intermediateShape,
                                const BaseDescriptor& descriptor,
                                DType dataType = DType_FP32)
{
    uint32_t numInputs = static_cast<uint32_t>(inputShape.size());
    uint32_t numOutputs = static_cast<uint32_t>(outputShape.size());

    std::string blockStr = TosaOpToString(Op_AVG_POOL2D) + "_block_";
    CHECK(basicBlock->GetName().find(blockStr)  != std::string::npos);
    CHECK(basicBlock->GetInputs().size() == numInputs);
    CHECK(basicBlock->GetOutputs().size() == numOutputs);
    CHECK(basicBlock->GetOperators().size() == 2);
    CHECK(basicBlock->GetTensors().size() == 3);

    //
    // Verify padding operator first.
    //

    TosaSerializationOperator* padOp = basicBlock->GetOperators().at(0);
    uint32_t padOpOutputs = 1;
    CHECK(padOp->GetInputTensorNames().size() == numInputs);
    CHECK(padOp->GetOutputTensorNames().size() == padOpOutputs);

    for (uint32_t i = 0; i < numInputs; i++)
    {
        std::basic_string<char> blockInputName = basicBlock->GetInputs()[i];
        std::basic_string<char> operatorInputName  = padOp->GetInputTensorNames()[i];

        std::string opStr = "input" + std::to_string(i) + "_";

        CHECK(blockInputName == operatorInputName);
        CHECK(basicBlock->GetTensorByName(blockInputName));
        CHECK(blockInputName.find(opStr)  != std::string::npos);

        TosaSerializationTensor* tensor = basicBlock->GetTensorByName(operatorInputName);
        CHECK(tensor->GetDtype() == dataType);
        CHECK(tensor->GetData().size() == 0);
        CHECK(tensor->GetShape() == inputShape[static_cast<unsigned long int>(i)]);
    }

    for (uint32_t i = 0; i < padOpOutputs; i++)
    {
        std::basic_string<char> operatorOutputName  = padOp->GetOutputTensorNames()[i];
        std::string opStr = "intermediate" + std::to_string(i) + "_";

        CHECK(basicBlock->GetTensorByName(operatorOutputName));
        CHECK(operatorOutputName.find(opStr)  != std::string::npos);

        TosaSerializationTensor* tensor = basicBlock->GetTensorByName(operatorOutputName);
        CHECK(tensor->GetDtype() == dataType);
        CHECK(tensor->GetData().size() == 0);
        CHECK(tensor->GetShape() == intermediateShape[static_cast<unsigned long int>(i)]);
    }

    CHECK(padOp->GetAttributeType() == Attribute_PadAttribute);
    CHECK(padOp->GetOp() == Op_PAD);

    VerifyTosaAttribute(descriptor,
                        padOp->GetAttribute(),
                        inputShape[0],
                        outputShape[0],
                        LayerType::Pooling2d);

    //
    // Verify average pool operator second.
    //

    TosaSerializationOperator* poolOp = basicBlock->GetOperators().at(1);
    uint32_t poolOpInputs = 1;
    CHECK(poolOp->GetInputTensorNames().size() == poolOpInputs);
    CHECK(poolOp->GetOutputTensorNames().size() == numOutputs);

    for (uint32_t i = 0; i < poolOpInputs; i++)
    {
        std::basic_string<char> operatorInputName  = poolOp->GetInputTensorNames()[i];
        std::string opStr = "intermediate" + std::to_string(i) + "_";

        CHECK(basicBlock->GetTensorByName(operatorInputName));
        CHECK(operatorInputName.find(opStr)  != std::string::npos);

        TosaSerializationTensor* tensor = basicBlock->GetTensorByName(operatorInputName);
        CHECK(tensor->GetDtype() == dataType);
        CHECK(tensor->GetData().size() == 0);
        CHECK(tensor->GetShape() == intermediateShape[static_cast<unsigned long int>(i)]);
    }

    for (uint32_t i = 0; i < numOutputs; i++)
    {
        std::basic_string<char> blockOutputName = basicBlock->GetOutputs()[i];
        std::basic_string<char> operatorOutputName  = poolOp->GetOutputTensorNames()[i];

        std::string opStr = "output" + std::to_string(i) + "_";

        CHECK(blockOutputName == operatorOutputName);
        CHECK(basicBlock->GetTensorByName(blockOutputName));
        CHECK(blockOutputName.find(opStr)  != std::string::npos);

        TosaSerializationTensor* tensor = basicBlock->GetTensorByName(operatorOutputName);
        CHECK(tensor->GetDtype() == dataType);
        CHECK(tensor->GetData().size() == 0);
        CHECK(tensor->GetShape() == outputShape[static_cast<unsigned long int>(i)]);
    }

    CHECK(poolOp->GetAttributeType() == Attribute_PoolAttribute);
    CHECK(poolOp->GetOp() == Op_AVG_POOL2D);

    VerifyTosaAttribute(descriptor,
                        poolOp->GetAttribute(),
                        inputShape[0],
                        outputShape[0],
                        LayerType::Pooling2d,
                        1);

}