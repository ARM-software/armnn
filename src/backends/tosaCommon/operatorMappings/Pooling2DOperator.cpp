//
// Copyright © 2022-2024 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "Pooling2DOperator.hpp"

TosaSerializationBasicBlock* ConvertPooling2DToTosaOperator(const Layer* layer,
                                                            const std::vector<const TensorInfo*>& inputs,
                                                            const std::vector<const TensorInfo*>& outputs,
                                                            const Pooling2dDescriptor* poolDescriptor)
{
    std::string poolType = (poolDescriptor->m_PoolType == PoolingAlgorithm::Max) ? "Op_MAX" : "Op_AVG";
    Op opcode = (poolDescriptor->m_PoolType == PoolingAlgorithm::Max) ? Op_MAX_POOL2D : Op_AVG_POOL2D;

    std::string inputName = std::string("input_");
    std::string outputName = std::string("output0_");
    std::string blockName  = std::string("Op_") + poolType + std::string("_POOL2D_block_") + GetUniqueTosaMappingID();

    // If a layer is present then the block will be used for execution, so input and output names need to be determined
    // using the previous and following layers so the graph is connected correctly. For validation this doesn't matter.
    if(layer != nullptr)
    {
        inputName = GenerateUniqueInputName(layer->GetInputSlot(0));
        outputName = GenerateUniqueOutputName(*layer);
    }

    DType accType = DType_INT32;
    DType inputDType0 = ArmNNToDType(inputs[0]->GetDataType());
    if (inputDType0 == DType_FP32)
    {
        accType = DType_FP32;
    }
    else if (inputDType0 == DType_FP16)
    {
        accType = DType_FP16;
    }

    std::vector<TosaSerializationTensor*> tensors;
    std::vector<TosaSerializationOperator*> operators;

    // Only add input tensors if connected layer is an input layer.
    // As intermediate or constant tensors will be created separately.
    // There also can't be duplicate tensor.
    std::vector<int32_t> inputShape0 = GetTosaTensorShape(inputs[0]->GetShape());
    if(inputName.find("input_") != std::string::npos)
    {
        tensors.push_back(new TosaSerializationTensor(inputName, inputShape0, inputDType0, {}));
    }

    std::vector<int32_t> outputShape0 = GetTosaTensorShape(outputs[0]->GetShape());
    DType outputDType0 = ArmNNToDType(outputs[0]->GetDataType());

    tensors.push_back(new TosaSerializationTensor(outputName, outputShape0, outputDType0, {}));

    std::vector<int> pad = {static_cast<int>(poolDescriptor->m_PadTop),
                            static_cast<int>(poolDescriptor->m_PadBottom),
                            static_cast<int>(poolDescriptor->m_PadLeft),
                            static_cast<int>(poolDescriptor->m_PadRight)};
    std::vector<int> kernel = {static_cast<int>(poolDescriptor->m_PoolHeight),
                               static_cast<int>(poolDescriptor->m_PoolWidth)};
    std::vector<int> stride = {static_cast<int>(poolDescriptor->m_StrideY),
                               static_cast<int>(poolDescriptor->m_StrideX)};
    std::vector<int> dilation = {1, 1};

    std::string sliceOutputName = GetInputSlicedToItsUsedSize(inputShape0,
                                                              inputName,
                                                              poolDescriptor->m_DataLayout,
                                                              inputDType0,
                                                              kernel,
                                                              pad,
                                                              stride,
                                                              dilation,
                                                              tensors,
                                                              operators,
                                                              true);

    TosaPoolAttribute attribute(pad, kernel, stride, 0, 0, accType);

    operators.push_back(new TosaSerializationOperator(opcode,
                                                      Attribute_PoolAttribute,
                                                      &attribute,
                                                      {sliceOutputName},
                                                      {outputName}));

    // operatorInputNames/operatorOutputNames ends up being the same as
    // blockInputNames/blockOutputNames for one-to-one ArmNN to TOSA mappings
    return new TosaSerializationBasicBlock(blockName, // name
                                           mainName, // region name
                                           operators, // operators
                                           tensors, // tensors
                                           {sliceOutputName}, // inputs
                                           {outputName}); // outputs
}