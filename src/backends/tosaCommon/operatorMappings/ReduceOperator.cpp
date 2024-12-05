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

    if (reduceDescriptor->m_vAxis.empty())
    {
        throw armnn::Exception("ConvertReduceOperator: Reduce Operation with empty axis not implemented.");
    }

    // Tensor names
    std::string inputName           = "input_";

    std::size_t intermediateCounter = 0;

    std::string outputName          = "output0_";

    std::string reduceOpName        = GetReduceOperationAsCString(reduceDescriptor->m_ReduceOperation);
    std::string blockName           = "Op_REDUCE_" + reduceOpName + "_block_" + GetUniqueTosaMappingID();

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

    int64_t input_zp  = 0;
    int64_t output_zp = 0;

    double input_scale  = 1.0;
    double output_scale = 1.0;

    int32_t input_multiplier  = 1;
    int32_t output_multiplier = 1;

    int32_t input_shift  = 0;
    int32_t output_shift = 0;

    int64_t numElemsOnReducedAxis = 1;

    std::vector<int32_t> axes(reduceDescriptor->m_vAxis.begin(), reduceDescriptor->m_vAxis.end());

    for (int64_t axis : axes)
    {
        numElemsOnReducedAxis *= inputShape[static_cast<uint64_t>(axis)];
    }

    std::vector<TosaSerializationOperator*> operators;

    bool inputQuantised = inputs[0]->IsQuantized();

    // Conditional RESCALE
    if (inputQuantised)
    {
        input_zp  = inputs[0]->GetQuantizationOffset();
        output_zp = outputs[0]->GetQuantizationOffset();

        std::string outputNameRescale =
            "intermediate" + std::to_string(intermediateCounter++) + "_" + GetUniqueTosaMappingID();

        TosaSerializationOperator* rescaleOp1 = nullptr;

        switch(reduceDescriptor->m_ReduceOperation)
        {
            case ReduceOperation::Sum:
                input_shift = 20;

                input_scale  = static_cast<double>(1 << input_shift) * inputs[0]->GetQuantizationScale();
                output_scale = 1.0 / (outputs[0]->GetQuantizationScale() * static_cast<double>(1 << input_shift));

                CreateRescaleTosaOperator(inputName,
                                          outputNameRescale,
                                          input_scale,
                                          static_cast<int32_t>(input_zp),
                                          0,
                                          false,
                                          false,
                                          true,
                                          true,
                                          &rescaleOp1);

                break;
            case ReduceOperation::Mean:
            {
                // calculate shifts and multipliers
                ComputeMultiplierAndShiftTosaScale32(1.0, input_multiplier, input_shift);
                ComputeMultiplierAndShiftTosaScale32
                (
                    static_cast<double>(inputs[0]->GetQuantizationScale()) /
                    static_cast<double>(outputs[0]->GetQuantizationScale()),
                    output_multiplier,
                    output_shift
                );

                int shift = 63 - __builtin_clzl(static_cast<uint64_t>(numElemsOnReducedAxis));
                shift = std::min(shift, 32);
                shift = std::min(shift, 62 - output_shift);

                output_multiplier = static_cast<int32_t>(
                                        (static_cast<int64_t>(output_multiplier) << shift) / numElemsOnReducedAxis);

                output_shift += shift;

                CreateRawRescaleTosaOperator(inputName,
                                             outputNameRescale,
                                             {input_multiplier},
                                             {input_shift},
                                             static_cast<int32_t>(input_zp),
                                             0,
                                             false,
                                             false,
                                             true,
                                             true,
                                             false,
                                             &rescaleOp1);
                break;
            }
            default:
                throw armnn::Exception("ConvertReduceOperator: Reduce Operation not implemented.");
        }

        operators.emplace_back(rescaleOp1);

        tensors.emplace_back(new TosaSerializationTensor(outputNameRescale,
                                                         inputShape,
                                                         DType_INT32,
                                                         {}));
    }

    // REDUCE_SUM
    for (const auto axis : axes)
    {
        auto rank = static_cast<int64_t>(inputs[0]->GetNumDimensions());

        if (axis < 0 || axis >= rank)
        {
            throw armnn::Exception("Axis value not within range of input shape.");
        }

        TosaAxisAttribute reduceAttribute(axis);

        std::string outputNameReduce =
            "intermediate" + std::to_string(intermediateCounter++) + "_" + GetUniqueTosaMappingID();

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

        std::vector<int32_t> outputShapeReduce = tensors.back()->GetShape();
        outputShapeReduce[static_cast<std::size_t>(axis)] = 1;

        tensors.emplace_back(new TosaSerializationTensor(outputNameReduce,
                                                         outputShapeReduce,
                                                         tensors.back()->GetDtype(),
                                                         {}));
    }

    // Conditional RESCALE
    if (inputQuantised)
    {
        std::string outputNameRescale =
            "intermediate" + std::to_string(intermediateCounter++) + "_" + GetUniqueTosaMappingID();

        TosaSerializationOperator* rescaleOp2 = nullptr;

        switch(reduceDescriptor->m_ReduceOperation)
        {
            case ReduceOperation::Sum:
                CreateRescaleTosaOperator(tensors.back()->GetName(),
                                          outputNameRescale,
                                          output_scale,
                                          0,
                                          static_cast<int32_t>(output_zp),
                                          false,
                                          false,
                                          true,
                                          true,
                                          &rescaleOp2);
                break;
            case ReduceOperation::Mean:
                CreateRawRescaleTosaOperator(tensors.back()->GetName(),
                                             outputNameRescale,
                                             {output_multiplier},
                                             {output_shift},
                                             0,
                                             static_cast<int32_t>(output_zp),
                                             false,
                                             false,
                                             true,
                                             true,
                                             false,
                                             &rescaleOp2);
                break;
            default:
                throw armnn::Exception("ConvertReduceOperator: Reduce Operation not implemented.");
        }

        operators.push_back(rescaleOp2);

        tensors.emplace_back(new TosaSerializationTensor(outputNameRescale,
                                                         tensors.back()->GetShape(),
                                                         inputType,
                                                         {}));
    }

    // RESHAPE
    TosaReshapeAttribute reshapeAttribute(GetTosaTensorShape(outputs[0]->GetShape()));

    std::string outputNameReshape = !inputQuantised && reduceDescriptor->m_ReduceOperation == ReduceOperation::Mean
        ? "intermediate" + std::to_string(intermediateCounter++) + "_" + GetUniqueTosaMappingID()
        : outputName;

    operators.emplace_back(new TosaSerializationOperator(Op_RESHAPE,
                                                         Attribute_ReshapeAttribute,
                                                         &reshapeAttribute,
                                                         { tensors.back()->GetName() },
                                                         { outputNameReshape }));

    tensors.emplace_back(new TosaSerializationTensor(outputNameReshape,
                                                     outputShape,
                                                     inputType,
                                                     {}));

    // Conditional MUL
    // Multiply previous tensor by constant of 1 / number of elements
    if (!inputQuantised && reduceDescriptor->m_ReduceOperation == ReduceOperation::Mean)
    {
        // Constant
        std::string constNameDivScale = "constant_" + GetUniqueTosaMappingID();
        inputNames.emplace_back(constNameDivScale);

        operators.push_back(new TosaSerializationOperator(Op_CONST,
                                                          Attribute_NONE,
                                                          nullptr,
                                                          {},
                                                          { constNameDivScale }));

        float divScale = 1.0f / static_cast<float>(numElemsOnReducedAxis);

        std::vector<uint8_t> uint8DivScale;
        switch (inputType)
        {
            case DType_FP32:
                TosaSerializationHandler::ConvertF32toU8({divScale}, uint8DivScale);
                break;
            case DType_FP16:
                TosaSerializationHandler::ConvertF16toU8({divScale}, uint8DivScale);
                break;
            default:
                throw armnn::Exception("ConvertReduceOperator: Data type not supported");
        }

        // Broadcast to match shapes
        std::vector<int32_t> divConstantShape(outputShape.size(), 1);

        tensors.push_back(new TosaSerializationTensor(constNameDivScale,
                                                      divConstantShape,
                                                      inputType,
                                                      uint8DivScale));

        // MUL
        TosaMulAttribute mulAttribute(0);

        operators.emplace_back(new TosaSerializationOperator(Op_MUL,
                                                             Attribute_MulAttribute,
                                                             &mulAttribute,
                                                             { constNameDivScale, outputNameReshape },
                                                             { outputName }));

        tensors.push_back(new TosaSerializationTensor(outputName,
                                                      outputShape,
                                                      inputType,
                                                      {}));
    }

    return new TosaSerializationBasicBlock(blockName,       // name
                                           mainName,        // region name
                                           operators,       // operators
                                           tensors,         // tensors
                                           inputNames,      // inputs
                                           { outputName }); // outputs
}
