//
// Copyright © 2024 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "DequantizeOperator.hpp"

TosaSerializationBasicBlock* ConvertDequantizeToTosaOperator(
    const Layer* layer,
    const std::vector<const TensorInfo*>& inputs,
    const std::vector<const TensorInfo*>& outputs)
{
    if (inputs.size() != 1)
    {
        throw Exception("ConvertDequantizeToTosaOperator: 1 input tensors required.");
    }

    if (outputs.size() != 1)
    {
        throw Exception("ConvertDequantizeToTosaOperator: 1 output tensor required.");
    }

    if (inputs[0]->HasPerAxisQuantization())
    {
        throw Exception("ConvertDequantizeToTosaOperator: Per axis quantization not currently supported.");
    }

    std::string inputName  = std::string("input_");
    std::string outputName = std::string("output_");
    std::string blockName  = std::string("Op_DEQUANTIZE_block_") + GetUniqueTosaMappingID();

    // If a layer is present then the block will be used for execution, so input and output names need to be determined
    // using the previous and following layers so the graph is connected correctly. For validation this doesn't matter.
    if (layer != nullptr)
    {
        inputName  = GenerateUniqueInputName(layer->GetInputSlot(0));
        outputName = GenerateUniqueOutputName(*layer);
    }

    std::vector<TosaSerializationTensor*> tensors;
    std::vector<TosaSerializationOperator*> operators;

    DataType inputDType = inputs[0]->GetDataType();
    DataType outputDType = outputs[0]->GetDataType();
    std::vector<int32_t> inputShape = GetTosaTensorShape(inputs[0]->GetShape());
    std::vector<int32_t> outputShape = GetTosaTensorShape(outputs[0]->GetShape());

    // Only add input tensors if connected layer is an input layer.
    // As intermediate or constant tensors will be created separately.
    // There also can't be duplicate tensor.
    if(inputName.find("input_") != std::string::npos)
    {
        tensors.push_back(new TosaSerializationTensor(inputName, inputShape, ArmNNToDType(inputDType), {}));
    }

    if (inputDType == DataType::Float16 ||
        inputDType == DataType::Float32)
    {
        operators.push_back(new TosaSerializationOperator(tosa::Op_CAST,
                                                          Attribute_NONE,
                                                          nullptr,
                                                          {inputName},
                                                          {outputName}));
    }
    else if (inputDType == DataType::QAsymmS8 ||
             inputDType == DataType::QSymmS16 ||
             inputDType == DataType::QSymmS8)
    {
        std::string outputNameCast      = std::string("intermediate0_") + GetUniqueTosaMappingID();
        std::string outputNameZeroPoint = std::string("constant0_") + GetUniqueTosaMappingID();
        std::string outputNameSub       = std::string("intermediate2_") + GetUniqueTosaMappingID();
        std::string outputNameScale     = std::string("constant1_") + GetUniqueTosaMappingID();

        float zeroPoint = static_cast<float>(inputs[0]->GetQuantizationOffset());
        float scale = inputs[0]->GetQuantizationScale();

        // cast
        TosaSerializationOperator* castOp = new TosaSerializationOperator(Op_CAST,
                                                                          Attribute_NONE,
                                                                          nullptr,
                                                                          {inputName},
                                                                          {outputNameCast});
        operators.push_back(castOp);
        tensors.push_back(new TosaSerializationTensor(outputNameCast, outputShape, ArmNNToDType(outputDType), {}));

        // const_zeroPoint
        TosaSerializationOperator* zeroPointOp = nullptr;
        TosaSerializationTensor* zeroPointTensor = nullptr;
        CreateConstTosaOperator<float>(outputNameZeroPoint,
                                       zeroPoint,
                                       ArmNNToDType(outputDType),
                                       inputShape,
                                       zeroPointOp,
                                       zeroPointTensor);
        operators.push_back(zeroPointOp);
        tensors.push_back(zeroPointTensor);

        // sub
        TosaSerializationOperator* subOp = new TosaSerializationOperator(Op_SUB,
                                                                         Attribute_NONE,
                                                                         nullptr,
                                                                         {outputNameCast, outputNameZeroPoint},
                                                                         {outputNameSub});
        operators.push_back(subOp);
        tensors.push_back(new TosaSerializationTensor(outputNameSub, outputShape, ArmNNToDType(outputDType), {}));

        // const_scale
        TosaSerializationOperator *scaleOp = nullptr;
        TosaSerializationTensor* scaleTensor = nullptr;
        CreateConstTosaOperator<float>(outputNameScale,
                                       scale,
                                       ArmNNToDType(outputDType),
                                       inputShape,
                                       scaleOp,
                                       scaleTensor);
        operators.push_back(scaleOp);
        tensors.push_back(scaleTensor);

        // mul
        int32_t shift = 0;
        TosaMulAttribute mulAttribute(shift);
        TosaSerializationOperator* mulOp = new TosaSerializationOperator(Op_MUL,
                                                                         Attribute_MulAttribute,
                                                                         &mulAttribute,
                                                                         {outputNameSub, outputNameScale},
                                                                         {outputName});
        operators.push_back(mulOp);
    }
    else
    {
        throw armnn::Exception("ConvertDequantizeToTosaOperator: Unsupported datatype."
                               " Only floating-point and signed quantized datatypes are supported.");
    }

    tensors.push_back(new TosaSerializationTensor(outputName, outputShape, ArmNNToDType(outputDType), {}));

    // operatorInputNames/operatorOutputNames ends up being the same as
    // blockInputNames/blockOutputNames for one-to-one ArmNN to Tosa mappings
    return new TosaSerializationBasicBlock(blockName,      // name
                                           mainName,       // region name
                                           operators,      // operators
                                           tensors,        // tensors
                                           {inputName},    // inputs
                                           {outputName});  // outputs
}