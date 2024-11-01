//
// Copyright © 2024 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
//
// Copyright © 2020 The TensorFlow Authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
//

#include "SoftmaxOperator.hpp"

#include "TosaRescaleOperatorUtils.hpp"
#include "TosaSoftmaxOperatorUtils.hpp"

// This function is paraphrased from:
// tensorflow/compiler/mlir/tosa/transforms/legalize_common.cc from function convertSoftmaxOp
TosaSerializationBasicBlock* ConvertSoftmaxToTosaOperator(const Layer* layer,
                                                          const std::vector<const TensorInfo*>& inputs,
                                                          const std::vector<const TensorInfo*>& outputs,
                                                          const SoftmaxDescriptor* softmaxDescriptor)

{
    ARMNN_THROW_INVALIDARG_MSG_IF_FALSE(IsQuantized8BitType(inputs[0]->GetDataType()),
    "ConvertSoftmaxToTosaOperator: Softmax currently only supports Int8 Quantized inputs");

    ARMNN_THROW_INVALIDARG_MSG_IF_FALSE(inputs.size() == 1,
    "ConvertSoftmaxToTosaOperator: Softmax must have only one input");

    ARMNN_THROW_INVALIDARG_MSG_IF_FALSE(outputs.size() >= 1,
    "ConvertSoftmaxToTosaOperator: Softmax must have at least one output");

    std::string inputName = std::string("input_");
    std::string outputName = std::string("output0_");
    std::string blockName = std::string("Op_SOFTMAX_block_") + GetUniqueTosaMappingID();

    std::string inputNameConst1 = std::string("intermediate_constant1_") + GetUniqueTosaMappingID();
    std::string inputNameConst2 = std::string("intermediate_constant2_") + GetUniqueTosaMappingID();
    std::string inputNameConst3 = std::string("intermediate_constant3_") + GetUniqueTosaMappingID();
    std::string inputNameConst3a = std::string("intermediate_constant3a_") + GetUniqueTosaMappingID();
    std::string inputNameConst4 = std::string("intermediate_constant4_") + GetUniqueTosaMappingID();
    std::string inputNameConst5 = std::string("intermediate_constant5_") + GetUniqueTosaMappingID();
    std::string inputNameConst6 = std::string("intermediate_constant6_") + GetUniqueTosaMappingID();
    std::string inputNameConst7 = std::string("intermediate_constant7_") + GetUniqueTosaMappingID();
    std::string inputNameConst8 = std::string("intermediate_constant8_") + GetUniqueTosaMappingID();
    std::string inputNameConst8a = std::string("intermediate_constant8a_") + GetUniqueTosaMappingID();
    std::string inputNameConst8b = std::string("intermediate_constant8b_") + GetUniqueTosaMappingID();
    std::string inputNameConst9 = std::string("intermediate_constant9_") + GetUniqueTosaMappingID();
    std::string inputNameConst9a = std::string("intermediate_constant9a_") + GetUniqueTosaMappingID();
    std::string inputNameConst9b = std::string("intermediate_constant9b_") + GetUniqueTosaMappingID();
    std::string inputNameConst10 = std::string("intermediate_constant10_") + GetUniqueTosaMappingID();

    std::string outputNameRescale1 = std::string("intermediate0_") + GetUniqueTosaMappingID();
    std::string outputNameReduceMax1 = std::string("intermediate1_") + GetUniqueTosaMappingID();
    std::string outputNameSub1 = std::string("intermediate2_") + GetUniqueTosaMappingID();
    std::string outputNameRescale2 = std::string("intermediate3_") + GetUniqueTosaMappingID();
    std::string outputNameTable1 = std::string("intermediate4_") + GetUniqueTosaMappingID();
    std::string outputNameTable2 = std::string("intermediate5_") + GetUniqueTosaMappingID();
    std::string outputNameTable3 = std::string("intermediate6_") + GetUniqueTosaMappingID();
    std::string outputNameTable4 = std::string("intermediate7_") + GetUniqueTosaMappingID();
    std::string outputNameLogicalL1 = std::string("intermediate8_") + GetUniqueTosaMappingID();
    std::string outputNameLogicalL2 = std::string("intermediate9_") + GetUniqueTosaMappingID();
    std::string outputNameLogicalL3 = std::string("intermediate10_") + GetUniqueTosaMappingID();
    std::string outputNameArithmeticR1 = std::string("intermediate11_") + GetUniqueTosaMappingID();
    std::string outputNameAdd1 = std::string("intermediate12_") + GetUniqueTosaMappingID();
    std::string outputNameAdd2 = std::string("intermediate13_") + GetUniqueTosaMappingID();
    std::string outputNameAdd3 = std::string("intermediate14_") + GetUniqueTosaMappingID();
    std::string outputNameArithmeticR2 = std::string("intermediate15_") + GetUniqueTosaMappingID();
    std::string outputNameReduceSum1 = std::string("intermediate16_") + GetUniqueTosaMappingID();
    std::string outputNameCLZ1 = std::string("intermediate17_") + GetUniqueTosaMappingID();
    std::string outputNameSub2 = std::string("intermediate18_") + GetUniqueTosaMappingID();
    std::string outputNameLogicalL4 = std::string("intermediate19_") + GetUniqueTosaMappingID();
    std::string outputNameMul1 = std::string("intermediate20_") + GetUniqueTosaMappingID();
    std::string outputNameAdd4 = std::string("intermediate21_") + GetUniqueTosaMappingID();
    std::string outputNameMul2 = std::string("intermediate22_") + GetUniqueTosaMappingID();
    std::string outputNameSub3 = std::string("intermediate23_") + GetUniqueTosaMappingID();
    std::string outputNameMul3 = std::string("intermediate24_") + GetUniqueTosaMappingID();
    std::string outputNameMul4 = std::string("intermediate25_") + GetUniqueTosaMappingID();
    std::string outputNameAdd5 = std::string("intermediate26_") + GetUniqueTosaMappingID();
    std::string outputNameMul5 = std::string("intermediate27_") + GetUniqueTosaMappingID();
    std::string outputNameSub4 = std::string("intermediate28_") + GetUniqueTosaMappingID();
    std::string outputNameMul6 = std::string("intermediate29_") + GetUniqueTosaMappingID();
    std::string outputNameMul7 = std::string("intermediate30_") + GetUniqueTosaMappingID();
    std::string outputNameAdd6 = std::string("intermediate31_") + GetUniqueTosaMappingID();
    std::string outputNameMul8 = std::string("intermediate32_") + GetUniqueTosaMappingID();
    std::string outputNameSub5 = std::string("intermediate33_") + GetUniqueTosaMappingID();
    std::string outputNameMul9 = std::string("intermediate34_") + GetUniqueTosaMappingID();
    std::string outputNameMul10 = std::string("intermediate35_") + GetUniqueTosaMappingID();
    std::string outputNameAdd7 = std::string("intermediate36_") + GetUniqueTosaMappingID();
    std::string outputNameMul11 = std::string("intermediate37_") + GetUniqueTosaMappingID();
    std::string outputNameSub6 = std::string("intermediate38_") + GetUniqueTosaMappingID();
    std::string outputNameArithmeticR3 = std::string("intermediate39_") + GetUniqueTosaMappingID();

    // If a layer is present then the block will be used for execution, so input and output names need to be determined
    // using the previous and following layers so the graph is connected correctly. For validation this doesn't matter.
    if (layer != nullptr)
    {
        inputName = GenerateUniqueInputName(layer->GetInputSlot(0));
        outputName = GenerateUniqueOutputName(*layer);
    }

    const TensorInfo& inputInfo = *inputs[0];
    const TensorInfo& outputInfo = *outputs[0];

    std::vector<TosaSerializationTensor *> tensors;
    std::vector<TosaSerializationOperator *> operators;

    std::vector<int32_t> inputShape0 = GetTosaTensorShape(inputInfo.GetShape());
    std::vector<int32_t> outputShape0 = GetTosaTensorShape(outputInfo.GetShape());

    DType inputDType0 = ArmNNToDType(inputInfo.GetDataType());
    DType outputDType0 = ArmNNToDType(outputInfo.GetDataType());

    // Only add input tensors if connected layer is an input layer.
    // As intermediate or constant tensors will be created separately.
    // There also can't be duplicate tensor.
    if (inputName.find("input_") != std::string::npos)
    {
        tensors.push_back(new TosaSerializationTensor(inputName, inputShape0, inputDType0, {}));
    }

    float inScale = inputInfo.GetQuantizationScale();

    int32_t input_zp = inputInfo.GetQuantizationOffset();
    int32_t output_zp = outputInfo.GetQuantizationOffset();

    std::vector<uint8_t> uint8Data;

    unsigned int rank = inputInfo.GetNumDimensions();
    const std::vector<int32_t> singleValueShape(rank,1);
    auto axis = static_cast<int32_t>(rank - 1);
    TosaAxisAttribute tosaAxisAttribute(axis);

    // softmax calculations done using only the last tensor dimension
    std::vector<int32_t> reduceShape = inputShape0;
    reduceShape[static_cast<unsigned long>(axis)] = 1;

    TosaSerializationOperator *rescaleOp1 = nullptr;
    CreateRescaleTosaOperator(inputName, outputNameRescale1, 1.0f, input_zp, 0,
                              false, false, false, true, &rescaleOp1);

    tensors.push_back(new TosaSerializationTensor(outputNameRescale1, inputShape0, DType_INT32, {}));
    operators.push_back(rescaleOp1);

    auto *reduceMaxOp1 = new TosaSerializationOperator(Op_REDUCE_MAX,
                                                       Attribute_AxisAttribute,
                                                       &tosaAxisAttribute,
                                                       {outputNameRescale1},
                                                       {outputNameReduceMax1});
    tensors.push_back(new TosaSerializationTensor(outputNameReduceMax1, reduceShape, DType_INT32, {}));
    operators.push_back(reduceMaxOp1);

    auto *subOp1 = new TosaSerializationOperator(Op_SUB,
                                                 Attribute_NONE,
                                                 nullptr,
                                                 {outputNameRescale1, outputNameReduceMax1},
                                                 {outputNameSub1});
    tensors.push_back(new TosaSerializationTensor(outputNameSub1, inputShape0, DType_INT32, {}));
    operators.push_back(subOp1);

    TosaSerializationOperator *rescaleOp2 = nullptr;
    CreateRescaleTosaOperator(outputNameSub1, outputNameRescale2, 128.0f, 0, 0, false, false, false, true, &rescaleOp2);
    tensors.push_back(new TosaSerializationTensor(outputNameRescale2, inputShape0, DType_INT16, {}));
    operators.push_back(rescaleOp2);

    std::array<std::vector <int16_t>, 4> lookupTables;
    CalculateSoftmaxTableValues(softmaxDescriptor->m_Beta, inScale, lookupTables);

    const std::vector<int16_t> first = lookupTables[0];
    const std::vector<int16_t> table1(&first[0],&first[0]+513);
    const std::vector<int16_t> second = lookupTables[1];
    const std::vector<int16_t> table2(&second[0],&second[0]+513);
    const std::vector<int16_t> third = lookupTables[2];
    const std::vector<int16_t> table3(&third[0],&third[0]+513);
    const std::vector<int16_t> fourth = lookupTables[3];
    const std::vector<int16_t> table4(&fourth[0],&fourth[0]+513);

    TosaTableAttribute tosaTableAttribute1(table1);
    TosaTableAttribute tosaTableAttribute2(table2);
    TosaTableAttribute tosaTableAttribute3(table3);
    TosaTableAttribute tosaTableAttribute4(table4);

    auto* tableOp1 = new TosaSerializationOperator(Op_TABLE,
                                                   Attribute_TableAttribute,
                                                   &tosaTableAttribute1,
                                                   {outputNameRescale2},
                                                   {outputNameTable1});
    tensors.push_back(new TosaSerializationTensor(outputNameTable1, inputShape0, DType_INT32, {}));
    operators.push_back(tableOp1);

    auto* tableOp2 = new TosaSerializationOperator(Op_TABLE,
                                                   Attribute_TableAttribute,
                                                   &tosaTableAttribute2,
                                                   {outputNameRescale2},
                                                   {outputNameTable2});
    tensors.push_back(new TosaSerializationTensor(outputNameTable2, inputShape0, DType_INT32, {}));
    operators.push_back(tableOp2);

    auto* tableOp3 = new TosaSerializationOperator(Op_TABLE,
                                                   Attribute_TableAttribute,
                                                   &tosaTableAttribute3,
                                                   {outputNameRescale2},
                                                   {outputNameTable3});
    tensors.push_back(new TosaSerializationTensor(outputNameTable3, inputShape0, DType_INT32, {}));
    operators.push_back(tableOp3);

    auto* tableOp4 = new TosaSerializationOperator(Op_TABLE,
                                                   Attribute_TableAttribute,
                                                   &tosaTableAttribute4,
                                                   {outputNameRescale2},
                                                   {outputNameTable4});
    tensors.push_back(new TosaSerializationTensor(outputNameTable4, inputShape0, DType_INT32, {}));
    operators.push_back(tableOp4);

    TosaSerializationHandler::ConvertI32toU8({17}, uint8Data);
    tensors.push_back(new TosaSerializationTensor(inputNameConst1,singleValueShape, DType_INT32,uint8Data));
    operators.push_back(new TosaSerializationOperator(Op_CONST, Attribute_NONE, nullptr, {}, {inputNameConst1}));

    auto* logicalLOp1 = new TosaSerializationOperator(Op_LOGICAL_LEFT_SHIFT,
                                                      Attribute_NONE,
                                                      nullptr,
                                                      {outputNameTable1, inputNameConst1},
                                                      {outputNameLogicalL1});
    tensors.push_back(new TosaSerializationTensor(outputNameLogicalL1, inputShape0, DType_INT32, {}));
    operators.push_back(logicalLOp1);

    TosaSerializationHandler::ConvertI32toU8({9}, uint8Data);
    tensors.push_back(new TosaSerializationTensor(inputNameConst2, singleValueShape, DType_INT32,uint8Data));
    operators.push_back(new TosaSerializationOperator(Op_CONST, Attribute_NONE, nullptr, {}, {inputNameConst2}));

    auto* logicalLOp2 = new TosaSerializationOperator(Op_LOGICAL_LEFT_SHIFT,
                                                      Attribute_NONE,
                                                      nullptr,
                                                      {outputNameTable2, inputNameConst2},
                                                      {outputNameLogicalL2});
    tensors.push_back(new TosaSerializationTensor(outputNameLogicalL2, inputShape0, DType_INT32, {}));
    operators.push_back(logicalLOp2);

    TosaSerializationHandler::ConvertI32toU8({1}, uint8Data);
    tensors.push_back(new TosaSerializationTensor(inputNameConst3, singleValueShape, DType_INT32,uint8Data));
    operators.push_back(new TosaSerializationOperator(Op_CONST, Attribute_NONE, nullptr, {}, {inputNameConst3}));

    auto* logicalLOp3 = new TosaSerializationOperator(Op_LOGICAL_LEFT_SHIFT,
                                                      Attribute_NONE,
                                                      nullptr,
                                                      {outputNameTable3, inputNameConst3},
                                                      {outputNameLogicalL3});
    tensors.push_back(new TosaSerializationTensor(outputNameLogicalL3, inputShape0, DType_INT32, {}));
    operators.push_back(logicalLOp3);

    bool rounding = true;
    TosaArithmeticRightShiftAttribute shiftRAttribute(rounding);

    TosaSerializationHandler::ConvertI32toU8({7}, uint8Data);
    tensors.push_back(new TosaSerializationTensor(inputNameConst4, singleValueShape, DType_INT32,uint8Data));
    operators.push_back(new TosaSerializationOperator(Op_CONST, Attribute_NONE, nullptr, {}, {inputNameConst4}));

    auto* arithmeticROp1 = new TosaSerializationOperator(Op_ARITHMETIC_RIGHT_SHIFT,
                                                         Attribute_ArithmeticRightShiftAttribute,
                                                         &shiftRAttribute,
                                                         {outputNameTable4, inputNameConst4},
                                                         {outputNameArithmeticR1});
    tensors.push_back(new TosaSerializationTensor(outputNameArithmeticR1, inputShape0, DType_INT32, {}));
    operators.push_back(arithmeticROp1);

    auto* addOp1 = new TosaSerializationOperator(Op_ADD,
                                                 Attribute_NONE,
                                                 nullptr,
                                                 {outputNameLogicalL1, outputNameLogicalL2},
                                                 {outputNameAdd1});
    tensors.push_back(new TosaSerializationTensor(outputNameAdd1, inputShape0, DType_INT32, {}));
    operators.push_back(addOp1);

    auto* addOp2 = new TosaSerializationOperator(Op_ADD,
                                                 Attribute_NONE,
                                                 nullptr,
                                                 {outputNameAdd1, outputNameLogicalL3},
                                                 {outputNameAdd2});
    tensors.push_back(new TosaSerializationTensor(outputNameAdd2, inputShape0, DType_INT32, {}));
    operators.push_back(addOp2);

    auto* addOp3 = new TosaSerializationOperator(Op_ADD,
                                                 Attribute_NONE,
                                                 nullptr,
                                                 {outputNameAdd2, outputNameArithmeticR1},
                                                 {outputNameAdd3});
    tensors.push_back(new TosaSerializationTensor(outputNameAdd3, inputShape0, DType_INT32, {}));
    operators.push_back(addOp3);

    TosaSerializationHandler::ConvertI32toU8({12}, uint8Data);
    tensors.push_back(new TosaSerializationTensor(inputNameConst5, singleValueShape, DType_INT32,uint8Data));
    operators.push_back(new TosaSerializationOperator(Op_CONST, Attribute_NONE, nullptr, {}, {inputNameConst5}));

    auto* arithmeticROp2 = new TosaSerializationOperator(Op_ARITHMETIC_RIGHT_SHIFT,
                                                         Attribute_ArithmeticRightShiftAttribute,
                                                         &shiftRAttribute,
                                                         {outputNameAdd3, inputNameConst5},
                                                         {outputNameArithmeticR2});
    tensors.push_back(new TosaSerializationTensor(outputNameArithmeticR2, inputShape0, DType_INT32, {}));
    operators.push_back(arithmeticROp2);

    auto* reduceSumOp1 = new TosaSerializationOperator(Op_REDUCE_SUM,
                                                       Attribute_AxisAttribute,
                                                       &tosaAxisAttribute,
                                                       {outputNameArithmeticR2},
                                                       {outputNameReduceSum1});
    tensors.push_back(new TosaSerializationTensor(outputNameReduceSum1, reduceShape, DType_INT32, {}));
    operators.push_back(reduceSumOp1);

    auto* countLeadingZeroOp1 = new TosaSerializationOperator(Op_CLZ,
                                                              Attribute_NONE,
                                                              nullptr,
                                                              {outputNameReduceSum1},
                                                              {outputNameCLZ1});
    tensors.push_back(new TosaSerializationTensor(outputNameCLZ1, reduceShape, DType_INT32, {}));
    operators.push_back(countLeadingZeroOp1);

    TosaSerializationHandler::ConvertI32toU8({1}, uint8Data);
    tensors.push_back(new TosaSerializationTensor(inputNameConst3a, singleValueShape, DType_INT32, uint8Data));
    operators.push_back(new TosaSerializationOperator(Op_CONST, Attribute_NONE, nullptr, {}, {inputNameConst3a}));

    auto* subOp2 = new TosaSerializationOperator(Op_SUB,
                                                 Attribute_NONE,
                                                 nullptr,
                                                 {outputNameCLZ1, inputNameConst3a},
                                                 {outputNameSub2});
    tensors.push_back(new TosaSerializationTensor(outputNameSub2, reduceShape, DType_INT32, {}));
    operators.push_back(subOp2);

    // half_denominator
    auto* logicalLOp4 = new TosaSerializationOperator(Op_LOGICAL_LEFT_SHIFT,
                                                      Attribute_NONE,
                                                      nullptr,
                                                      {outputNameReduceSum1, outputNameSub2},
                                                      {outputNameLogicalL4});
    tensors.push_back(new TosaSerializationTensor(outputNameLogicalL4, reduceShape, DType_INT32, {}));
    operators.push_back(logicalLOp4);

    TosaMulAttribute mulAttribute1(31);

    TosaSerializationHandler::ConvertI32toU8({-1010580540}, uint8Data); // constant_neg_32_over_17
    tensors.push_back(new TosaSerializationTensor(inputNameConst6, singleValueShape, DType_INT32,uint8Data));
    operators.push_back(new TosaSerializationOperator(Op_CONST, Attribute_NONE, nullptr, {}, {inputNameConst6}));

    // mul_half_denominator
    auto* mulOp1 = new TosaSerializationOperator(Op_MUL,
                                                 Attribute_MulAttribute,
                                                 &mulAttribute1,
                                                 {outputNameLogicalL4, inputNameConst6},
                                                 {outputNameMul1});
    tensors.push_back(new TosaSerializationTensor(outputNameMul1, reduceShape, DType_INT32, {}));
    operators.push_back(mulOp1);

    TosaSerializationHandler::ConvertI32toU8({1515870810}, uint8Data); // constant_48_over_17
    tensors.push_back(new TosaSerializationTensor(inputNameConst7, singleValueShape, DType_INT32,uint8Data));
    operators.push_back(new TosaSerializationOperator(Op_CONST, Attribute_NONE, nullptr, {}, {inputNameConst7}));

    // nr_x
    auto* addOp4 = new TosaSerializationOperator(Op_ADD,
                                                 Attribute_NONE,
                                                 nullptr,
                                                 {outputNameMul1, inputNameConst7},
                                                 {outputNameAdd4});
    tensors.push_back(new TosaSerializationTensor(outputNameAdd4, reduceShape, DType_INT32, {}));
    operators.push_back(addOp4);

    // Newton-Raphson 3 iterations of MUL SUB MUL MUL ADD sequence
    auto* mulOp2 = new TosaSerializationOperator(Op_MUL,
                                                 Attribute_MulAttribute,
                                                 &mulAttribute1,
                                                 {outputNameAdd4, outputNameLogicalL4},
                                                 {outputNameMul2});
    tensors.push_back(new TosaSerializationTensor(outputNameMul2, reduceShape, DType_INT32, {}));
    operators.push_back(mulOp2);

    TosaSerializationHandler::ConvertI32toU8({536870912}, uint8Data); // F2_one constant
    tensors.push_back(new TosaSerializationTensor(inputNameConst8, singleValueShape, DType_INT32,uint8Data));
    operators.push_back(new TosaSerializationOperator(Op_CONST, Attribute_NONE, nullptr, {}, {inputNameConst8}));

    auto* subOp3 = new TosaSerializationOperator(Op_SUB,
                                                 Attribute_NONE,
                                                 nullptr,
                                                 {inputNameConst8, outputNameMul2},
                                                 {outputNameSub3});
    tensors.push_back(new TosaSerializationTensor(outputNameSub3, reduceShape, DType_INT32, {}));
    operators.push_back(subOp3);

    auto* mulOp3 = new TosaSerializationOperator(Op_MUL,
                                                 Attribute_MulAttribute,
                                                 &mulAttribute1,
                                                 {outputNameAdd4, outputNameSub3},
                                                 {outputNameMul3});
    tensors.push_back(new TosaSerializationTensor(outputNameMul3, reduceShape, DType_INT32, {}));
    operators.push_back(mulOp3);

    TosaMulAttribute mulAttribute2(0);

    TosaSerializationHandler::ConvertI32toU8({4}, uint8Data);
    tensors.push_back(new TosaSerializationTensor(inputNameConst9, singleValueShape, DType_INT32,uint8Data));
    operators.push_back(new TosaSerializationOperator(Op_CONST, Attribute_NONE, nullptr, {}, {inputNameConst9}));

    auto* mulOp4 = new TosaSerializationOperator(Op_MUL,
                                                 Attribute_MulAttribute,
                                                 &mulAttribute2,
                                                 {outputNameMul3, inputNameConst9},
                                                 {outputNameMul4});
    tensors.push_back(new TosaSerializationTensor(outputNameMul4, reduceShape, DType_INT32, {}));
    operators.push_back(mulOp4);

    auto* addOp5 = new TosaSerializationOperator(Op_ADD,
                                                 Attribute_NONE,
                                                 nullptr,
                                                 {outputNameAdd4, outputNameMul4},
                                                 {outputNameAdd5});
    tensors.push_back(new TosaSerializationTensor(outputNameAdd5, reduceShape, DType_INT32, {}));
    operators.push_back(addOp5);

    // Newton-Raphson 2nd iteration... nr_x = op25_add_x_op24.getResult();
    auto* mulOp5 = new TosaSerializationOperator(Op_MUL,
                                                 Attribute_MulAttribute,
                                                 &mulAttribute1,
                                                 {outputNameAdd5, outputNameLogicalL4},
                                                 {outputNameMul5});
    tensors.push_back(new TosaSerializationTensor(outputNameMul5, reduceShape, DType_INT32, {}));
    operators.push_back(mulOp5);

    TosaSerializationHandler::ConvertI32toU8({536870912}, uint8Data);
    tensors.push_back(new TosaSerializationTensor(inputNameConst8a, singleValueShape, DType_INT32,uint8Data));
    operators.push_back(new TosaSerializationOperator(Op_CONST, Attribute_NONE, nullptr, {}, {inputNameConst8a}));

    auto* subOp4 = new TosaSerializationOperator(Op_SUB,
                                                 Attribute_NONE,
                                                 nullptr,
                                                 {inputNameConst8a, outputNameMul5},
                                                 {outputNameSub4});
    tensors.push_back(new TosaSerializationTensor(outputNameSub4, reduceShape, DType_INT32, {}));
    operators.push_back(subOp4);

    auto* mulOp6 = new TosaSerializationOperator(Op_MUL,
                                                 Attribute_MulAttribute,
                                                 &mulAttribute1,
                                                 {outputNameAdd5, outputNameSub4},
                                                 {outputNameMul6});
    tensors.push_back(new TosaSerializationTensor(outputNameMul6, reduceShape, DType_INT32, {}));
    operators.push_back(mulOp6);

    TosaSerializationHandler::ConvertI32toU8({4}, uint8Data);
    tensors.push_back(new TosaSerializationTensor(inputNameConst9a, singleValueShape, DType_INT32,uint8Data));
    operators.push_back(new TosaSerializationOperator(Op_CONST, Attribute_NONE, nullptr, {}, {inputNameConst9a}));

    auto* mulOp7 = new TosaSerializationOperator(Op_MUL,
                                                 Attribute_MulAttribute,
                                                 &mulAttribute2,
                                                 {outputNameMul6, inputNameConst9a},
                                                 {outputNameMul7});
    tensors.push_back(new TosaSerializationTensor(outputNameMul7, reduceShape, DType_INT32, {}));
    operators.push_back(mulOp7);

    auto* addOp6 = new TosaSerializationOperator(Op_ADD,
                                                 Attribute_NONE,
                                                 nullptr,
                                                 {outputNameAdd5, outputNameMul7},
                                                 {outputNameAdd6});
    tensors.push_back(new TosaSerializationTensor(outputNameAdd6, reduceShape, DType_INT32, {}));
    operators.push_back(addOp6);

    // Newton-Raphson 3rd iteration... nr_x = op25_add_x_op24.getResult();
    auto* mulOp8 = new TosaSerializationOperator(Op_MUL,
                                                 Attribute_MulAttribute,
                                                 &mulAttribute1,
                                                 {outputNameAdd6, outputNameLogicalL4},
                                                 {outputNameMul8});
    tensors.push_back(new TosaSerializationTensor(outputNameMul8, reduceShape, DType_INT32, {}));
    operators.push_back(mulOp8);

    TosaSerializationHandler::ConvertI32toU8({536870912}, uint8Data);
    tensors.push_back(new TosaSerializationTensor(inputNameConst8b, singleValueShape, DType_INT32,uint8Data));
    operators.push_back(new TosaSerializationOperator(Op_CONST, Attribute_NONE, nullptr, {}, {inputNameConst8b}));

    auto* subOp5 = new TosaSerializationOperator(Op_SUB,
                                                 Attribute_NONE,
                                                 nullptr,
                                                 {inputNameConst8b, outputNameMul8},
                                                 {outputNameSub5});
    tensors.push_back(new TosaSerializationTensor(outputNameSub5, reduceShape, DType_INT32, {}));
    operators.push_back(subOp5);

    auto* mulOp9 = new TosaSerializationOperator(Op_MUL,
                                                 Attribute_MulAttribute,
                                                 &mulAttribute1,
                                                 {outputNameAdd6, outputNameSub5},
                                                 {outputNameMul9});
    tensors.push_back(new TosaSerializationTensor(outputNameMul9, reduceShape, DType_INT32, {}));
    operators.push_back(mulOp9);

    TosaSerializationHandler::ConvertI32toU8({4}, uint8Data);
    tensors.push_back(new TosaSerializationTensor(inputNameConst9b, singleValueShape, DType_INT32,uint8Data));
    operators.push_back(new TosaSerializationOperator(Op_CONST, Attribute_NONE, nullptr, {}, {inputNameConst9b}));

    auto* mulOp10 = new TosaSerializationOperator(Op_MUL,
                                                  Attribute_MulAttribute,
                                                  &mulAttribute2,
                                                  {outputNameMul9, inputNameConst9b},
                                                  {outputNameMul10});
    tensors.push_back(new TosaSerializationTensor(outputNameMul10, reduceShape, DType_INT32, {}));
    operators.push_back(mulOp10);

    auto* addOp7 = new TosaSerializationOperator(Op_ADD,
                                                 Attribute_NONE,
                                                 nullptr,
                                                 {outputNameAdd6, outputNameMul10},
                                                 {outputNameAdd7});
    tensors.push_back(new TosaSerializationTensor(outputNameAdd7, reduceShape, DType_INT32, {}));
    operators.push_back(addOp7);

    TosaMulAttribute mulAttribute3(30);

    auto* mulOp11 = new TosaSerializationOperator(Op_MUL,
                                                  Attribute_MulAttribute,
                                                  &mulAttribute3,
                                                  {outputNameAdd3, outputNameAdd7},
                                                  {outputNameMul11});
    tensors.push_back(new TosaSerializationTensor(outputNameMul11, outputShape0, DType_INT32, {}));
    operators.push_back(mulOp11);

    TosaSerializationHandler::ConvertI32toU8({35}, uint8Data);
    tensors.push_back(new TosaSerializationTensor(inputNameConst10, singleValueShape, DType_INT32,uint8Data));
    operators.push_back(new TosaSerializationOperator(Op_CONST, Attribute_NONE, nullptr, {}, {inputNameConst10}));

    auto* subOp6 = new TosaSerializationOperator(Op_SUB,
                                                 Attribute_NONE,
                                                 nullptr,
                                                 {inputNameConst10, outputNameCLZ1},
                                                 {outputNameSub6});
    tensors.push_back(new TosaSerializationTensor(outputNameSub6, reduceShape, DType_INT32, {}));
    operators.push_back(subOp6);

    auto* arithmeticROp3 = new TosaSerializationOperator(Op_ARITHMETIC_RIGHT_SHIFT,
                                                         Attribute_ArithmeticRightShiftAttribute,
                                                         &shiftRAttribute,
                                                         {outputNameMul11, outputNameSub6},
                                                         {outputNameArithmeticR3});
    tensors.push_back(new TosaSerializationTensor(outputNameArithmeticR3, outputShape0, DType_INT32, {}));
    operators.push_back(arithmeticROp3);

    TosaSerializationOperator* rescaleOp3 = nullptr;
    CreateRescaleTosaOperator(outputNameArithmeticR3, outputName, 1.0f, 0, output_zp, false, false, 
                              false, true, &rescaleOp3);

    tensors.push_back(new TosaSerializationTensor(outputName, outputShape0, outputDType0, {}));
    operators.push_back(rescaleOp3);

    return new TosaSerializationBasicBlock(blockName,
                                           mainName,
                                           {operators},
                                           tensors,
                                           {inputName},
                                           {outputName});
}