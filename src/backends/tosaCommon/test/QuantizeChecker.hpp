//
// Copyright © 2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "TosaTestUtils.hpp"

using namespace armnn;
using namespace tosa;

void VerifyQuantize(TosaSerializationBasicBlock* quantizeBlock,
                    std::vector<int32_t> shape,
                    DType inputDataType = DType_FP32,
                    DType outputDataType = DType_FP32)
{
    std::string blockStr = "Op_QUANTIZE_block_";
    CHECK(quantizeBlock->GetName().find(blockStr)  != std::string::npos);
    CHECK(quantizeBlock->GetInputs().size() == 1);
    CHECK(quantizeBlock->GetOutputs().size() == 1);
    CHECK(quantizeBlock->GetOperators().size() == 5); // MUL, CONST, ADD, CONST, CAST
    CHECK(quantizeBlock->GetTensors().size() == 6);

    std::basic_string<char> blockInputName = quantizeBlock->GetInputs()[0];
    std::basic_string<char> blockOutputName = quantizeBlock->GetOutputs()[0];

    //
    // Verify Constants
    //
    TosaSerializationOperator* constZeroPointOp = quantizeBlock->GetOperators().at(0);
    CHECK(constZeroPointOp->GetAttributeType() == Attribute_NONE);
    CHECK(constZeroPointOp->GetOp() == tosa::Op_CONST);

    TosaSerializationOperator* constScaleOp = quantizeBlock->GetOperators().at(1);
    CHECK(constScaleOp->GetAttributeType() == Attribute_NONE);
    CHECK(constScaleOp->GetOp() == tosa::Op_CONST);

    //
    // Verify Multiplication
    //
    ElementwiseBinaryDescriptor mulDescriptor(BinaryOperation::Mul);
    TosaSerializationOperator* mulOp = quantizeBlock->GetOperators().at(2);
    CHECK(mulOp->GetAttributeType() == tosa::Attribute_MulAttribute);
    CHECK(mulOp->GetOp() == tosa::Op_MUL);

    CHECK(mulOp->GetInputTensorNames().size() == 2);
    std::basic_string<char> mulInputName0 = mulOp->GetInputTensorNames()[0];
    std::basic_string<char> mulInputName1 = mulOp->GetInputTensorNames()[1];

    CHECK(blockInputName == mulInputName0);

    TosaSerializationTensor* mulInputTensor0 = quantizeBlock->GetTensorByName(mulInputName0);
    CHECK(mulInputTensor0->GetDtype() == inputDataType);
    CHECK(mulInputTensor0->GetData().size() == 0);
    CHECK(mulInputTensor0->GetShape() == shape);

    TosaSerializationTensor* mulInputTensor1 = quantizeBlock->GetTensorByName(mulInputName1);
    CHECK(mulInputTensor1->GetShape() == shape);

    //
    // Verify Addition
    //
    ElementwiseBinaryDescriptor addDescriptor(BinaryOperation::Add);
    TosaSerializationOperator* addOp = quantizeBlock->GetOperators().at(3);
    CHECK(addOp->GetAttributeType() == Attribute_NONE);
    CHECK(addOp->GetOp() == tosa::Op_ADD);

    CHECK(addOp->GetInputTensorNames().size() == 2);
    std::basic_string<char> addInputName0 = addOp->GetInputTensorNames()[0];
    std::basic_string<char> addInputName1 = addOp->GetInputTensorNames()[1];

    TosaSerializationTensor* addInputTensor0 = quantizeBlock->GetTensorByName(addInputName0);
    CHECK(addInputTensor0->GetDtype() == inputDataType);
    CHECK(addInputTensor0->GetData().size() == 0);
    CHECK(addInputTensor0->GetShape() == shape);

    TosaSerializationTensor* addInputTensor1 = quantizeBlock->GetTensorByName(addInputName1);
    CHECK(addInputTensor1->GetShape() == shape);

    //
    // Verify Cast
    //
    TosaSerializationOperator* castOp = quantizeBlock->GetOperators().at(4);
    CHECK(castOp->GetAttributeType() == Attribute_NONE);
    CHECK(castOp->GetOp() == tosa::Op_CAST);

    CHECK(castOp->GetInputTensorNames().size() == 1);
    CHECK(castOp->GetOutputTensorNames().size() == 1);

    std::basic_string<char> castInputName = castOp->GetInputTensorNames()[0];
    std::basic_string<char> castOutputName = castOp->GetOutputTensorNames()[0];

    TosaSerializationTensor* castInputTensor = quantizeBlock->GetTensorByName(castInputName);
    CHECK(castInputTensor->GetDtype() == inputDataType);
    CHECK(castInputTensor->GetData().size() == 0);
    CHECK(castInputTensor->GetShape() == shape);

    TosaSerializationTensor* castOutputTensor = quantizeBlock->GetTensorByName(castOutputName);
    CHECK(castOutputTensor->GetDtype() == outputDataType);
    CHECK(castOutputTensor->GetData().size() == 0);
    CHECK(castOutputTensor->GetShape() == shape);

    CHECK(blockOutputName == castOutputName);


}