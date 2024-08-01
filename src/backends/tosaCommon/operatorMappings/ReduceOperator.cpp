//
// Copyright © 2024 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Copyright © 2020 The TensorFlow Authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
//

#include "ReduceOperator.hpp"

#include <armnn/TypesUtils.hpp>
#include "TosaRescaleOperatorUtils.hpp"

// This function is paraphrased from:
// tensorflow/compiler/mlir/tosa/transforms/legalize_common.cc from functions convertReduceMeanOp, convertReduceSumOp,
// convertReduceOpCommon
TosaSerializationBasicBlock* ConvertReduceToTosaOperator(const Layer* layer,
                                                         const std::vector<const TensorInfo*>& inputs,
                                                         const std::vector<const TensorInfo*>& outputs,
                                                         const ReduceDescriptor* reduceDescriptor)
{
    // Early exits
    if (!inputs[0])
    {
        throw armnn::Exception("ConvertReduceOperator: Must provide a valid input tensor.");
    }

    if (inputs[0]->IsQuantized() ^ outputs[0]->IsQuantized())
    {
        throw armnn::Exception("ConvertReduceOperator: "
                               "Both input and output tensors must be either quantised or non-quantised data types.");
    }

    if (reduceDescriptor->m_vAxis.size() > 1)
    {
        throw armnn::Exception("ConvertReduceOperator: Reduce Operation with multiple axes not implemented.");
    }

    if (reduceDescriptor->m_vAxis.empty())
    {
        throw armnn::Exception("ConvertReduceOperator: Reduce Operation with empty axis not implemented.");
    }

    auto axis = static_cast<int32_t>(reduceDescriptor->m_vAxis[0]);
    auto rank = static_cast<int32_t>(inputs[0]->GetNumDimensions());

    if (axis < 0 || axis >= rank)
    {
        throw armnn::Exception("Axis value not within range of input shape.");
    }

    // Tensor names
    std::string inputName          = "input_";

    std::string outputNameRescale1 = "intermediate0_" + GetUniqueTosaMappingID();
    std::string outputNameReduce   = "intermediate1_" + GetUniqueTosaMappingID();
    std::string outputNameRescale2 = "intermediate2_" + GetUniqueTosaMappingID();
    std::string outputNameMul      = "intermediate3_" + GetUniqueTosaMappingID();

    std::string outputName         = "output0_";

    std::string reduceOpName       = GetReduceOperationAsCString(reduceDescriptor->m_ReduceOperation);
    std::string blockName          = "Op_REDUCE_" + reduceOpName + "_block_" + GetUniqueTosaMappingID();

    std::vector<int32_t> inputShape  = GetTosaTensorShape(inputs[0]->GetShape());
    std::vector<int32_t> outputShape = GetTosaTensorShape(outputs[0]->GetShape());

    if (layer)
    {
        inputName  = GenerateUniqueInputName(layer->GetInputSlot(0));
        outputName = GenerateUniqueOutputName(*layer);
    }

    std::vector<TosaSerializationTensor*> tensors;
    std::vector<std::string> inputNames{inputName};

    DType inputType = ArmNNToDType(inputs[0]->GetDataType());

    if (inputName.substr(0, 6) == "input_")
    {
        tensors.emplace_back(new TosaSerializationTensor(inputName,
                                                         inputShape,
                                                         inputType,
                                                         {}));
    }

    int32_t input_shift = 20;

    double input_scale  = static_cast<double>(1 << input_shift) * inputs[0]->GetQuantizationScale();
    double output_scale = 1.0 / (outputs[0]->GetQuantizationScale() * static_cast<double>(1 << input_shift));

    int32_t input_zp    =  0;
    int32_t output_zp   =  0;

    std::vector<TosaSerializationOperator*> operators;

    // Conditional RESCALE
    if (inputs[0]->IsQuantized())
    {
        TosaSerializationOperator* rescaleOp1 = nullptr;

        CreateRescaleTosaOperator(inputName,
                                  outputNameRescale1,
                                  input_scale,
                                  input_zp,
                                  output_zp,
                                  true,
                                  true,
                                  &rescaleOp1);

        operators.emplace_back(rescaleOp1);

        tensors.emplace_back(new TosaSerializationTensor(outputNameRescale1,
                                                         inputShape,
                                                         DType_INT32,
                                                         {}));
    }

    // REDUCE
    TosaAxisAttribute reduceAttribute(axis);

    switch(reduceDescriptor->m_ReduceOperation)
    {
        case ReduceOperation::Sum:
        case ReduceOperation::Mean:
            operators.emplace_back(new TosaSerializationOperator(Op_REDUCE_SUM,
                                                                 Attribute_AxisAttribute,
                                                                 &reduceAttribute,
                                                                 { tensors.back()->GetName() },
                                                                 { outputNameReduce }));
            break;
        default:
            throw armnn::Exception("ConvertReduceOperator: Reduce Operation not implemented.");
    }

    std::vector<int32_t> outputShapeReduce = inputShape;
    outputShapeReduce[reduceDescriptor->m_vAxis[0]] = 1;

    tensors.emplace_back(new TosaSerializationTensor(outputNameReduce,
                                                     outputShapeReduce,
                                                     tensors.back()->GetDtype(),
                                                     {}));

    // Conditional RESCALE
    auto numElemsOnReducedAxis = inputShape[static_cast<unsigned long>(axis)];
    float divScale = 1.0f / static_cast<float>(numElemsOnReducedAxis);
    if (inputs[0]->IsQuantized())
    {

        // if Mean, modify output_scale to account for 1/number of elements.
        if (reduceDescriptor->m_ReduceOperation == ReduceOperation::Mean)
        {
            output_scale *= divScale;
        }

        TosaSerializationOperator* rescaleOp2 = nullptr;

        CreateRescaleTosaOperator(outputNameReduce,
                                  outputNameRescale2,
                                  output_scale,
                                  output_zp,
                                  input_zp,
                                  true,
                                  true,
                                  &rescaleOp2);

        operators.push_back(rescaleOp2);

        tensors.emplace_back(new TosaSerializationTensor(outputNameRescale2,
                                                         outputShapeReduce,
                                                         inputType,
                                                         {}));
    }
    else
    {
        // if Mean, add Mul to account for 1/number of elements.
        if (reduceDescriptor->m_ReduceOperation == ReduceOperation::Mean)
        {
            // CONSTANT operator, value to multiply by
            std::string divConstantName = std::string("constant_") + GetUniqueTosaMappingID();
            inputNames.emplace_back(divConstantName);

            operators.push_back(new TosaSerializationOperator(Op_CONST,
                                                              Attribute_NONE,
                                                              nullptr,
                                                              {},
                                                              {divConstantName}));

            std::vector<uint8_t> uint8DivScale;
            switch (inputType)
            {
                case DType_FP16:
                    TosaSerializationHandler::ConvertF16toU8({divScale}, uint8DivScale);
                    break;
                case DType_FP32:
                    TosaSerializationHandler::ConvertF32toU8({divScale}, uint8DivScale);
                    break;
                default:
                    throw armnn::Exception("ConvertReduceOperator: Data type not supported");
            }

            std::vector<int32_t> divConstantShape (outputShapeReduce.size(), 1);
            tensors.push_back(new TosaSerializationTensor(divConstantName,
                                                          divConstantShape,
                                                          inputType,
                                                          uint8DivScale));

            // MUL operator
            int8_t shift = 0;
            TosaMulAttribute mulAttribute(shift);

            operators.emplace_back(new TosaSerializationOperator(Op_MUL,
                                                                 Attribute_MulAttribute,
                                                                 &mulAttribute,
                                                                 {divConstantName, outputNameReduce},
                                                                 {outputNameMul}));
            tensors.push_back(new TosaSerializationTensor(outputNameMul,
                                                          outputShapeReduce,
                                                          inputType,
                                                          {}));
        }
    }

    // RESHAPE
    TosaReshapeAttribute reshapeAttribute(GetTosaTensorShape(outputs[0]->GetShape()));

    operators.emplace_back(new TosaSerializationOperator(Op_RESHAPE,
                                                         Attribute_ReshapeAttribute,
                                                         &reshapeAttribute,
                                                         { tensors.back()->GetName() },
                                                         { outputName }));

    tensors.emplace_back(new TosaSerializationTensor(outputName,
                                                     outputShape,
                                                     inputType,
                                                     {}));

    return new TosaSerializationBasicBlock(blockName,       // name
                                           mainName,        // region name
                                           operators,       // operators
                                           tensors,         // tensors
                                           inputNames,      // inputs
                                           { outputName }); // outputs
}
