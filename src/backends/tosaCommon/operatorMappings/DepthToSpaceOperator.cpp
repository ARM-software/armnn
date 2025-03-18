//
// Copyright © 2024 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
//
// Copyright © 2020 The TensorFlow Authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
//

#include "DepthToSpaceOperator.hpp"

// This function is paraphrased from:
// tensorflow/compiler/mlir/tosa/transforms/legalize_tfl.cc from function ConvertTFLDepthToSpaceOp
TosaSerializationBasicBlock* ConvertDepthToSpaceToTosaOperator(const Layer* layer,
                                                               const std::vector<const TensorInfo*>& inputs,
                                                               const std::vector<const TensorInfo*>& outputs,
                                                               const DepthToSpaceDescriptor* descriptor)
{
    // TOSA currently only supports NHWC input
    if (descriptor->m_DataLayout != DataLayout::NHWC)
    {
        throw InvalidArgumentException("Only NHWC input is supported for DepthToSpace");
    }

    // TOSA currently only supports 4D input
    if (inputs[0]->GetNumDimensions() != 4)
    {
        throw InvalidArgumentException("Only 4D input is supported for DepthToSpace");
    }

    std::string inputName;
    std::string outputName = std::string("output0_");
    std::string outputReshapeName = std::string("layer_intermediate0_") + GetUniqueTosaMappingID();
    std::string outputTransposeName = std::string("layer_intermediate1_") + GetUniqueTosaMappingID();
    std::string blockName  = std::string("Op_DEPTHTOSPACE_block_") + GetUniqueTosaMappingID();

    DType inputDType0 = ArmNNToDType(inputs[0]->GetDataType());
    DType outputDType0 = ArmNNToDType(outputs[0]->GetDataType());

    // Set input names for validation purposes only.
    if(layer == nullptr)
    {
        inputName = "input_0";
    }
    // If a layer is present then the block will be used for execution, so input and output names need to be
    // determined using the previous and following layers so the graph is connected correctly.
    // For validation this doesn't matter.
    else
    {
        // Get the layer connected to the input slot and determine unique tensor names.
        for (uint32_t i = 0; i < inputs.size(); ++i)
        {
            inputName = GenerateUniqueInputName(layer->GetInputSlot(i));
        }

        // Determine unique output tensor name.
        outputName = GenerateUniqueOutputName(*layer);
    }

    std::vector<TosaSerializationTensor*> tensors;
    std::vector<TosaSerializationOperator*> operators;

    // Setup input Tensor: Only add tensor if connected layer is an input layer.
    // As intermediate or constant tensors will be created separately. There also can't be duplicate tensors.
    if(inputName.find("input_") != std::string::npos)
    {
        std::vector<int32_t> inputShape0 = GetTosaTensorShape(inputs[0]->GetShape());
        tensors.push_back(new TosaSerializationTensor(inputName, inputShape0, inputDType0, {}));
    }

    std::vector<int32_t> reshapeDims1
    {
        static_cast<int32_t>(inputs[0]->GetShape()[0]),
        static_cast<int32_t>(inputs[0]->GetShape()[1]),
        static_cast<int32_t>(inputs[0]->GetShape()[2]),
        static_cast<int32_t>(descriptor->m_BlockSize),
        static_cast<int32_t>(descriptor->m_BlockSize),
        static_cast<int32_t>(inputs[0]->GetShape()[3] / (descriptor->m_BlockSize * descriptor->m_BlockSize))
    };
    TosaReshapeAttribute reshapeAttr1(reshapeDims1);
    auto* reshapeOp1 = new TosaSerializationOperator(Op_RESHAPE,
                                                     Attribute_ReshapeAttribute,
                                                     &reshapeAttr1,
                                                     {inputName},
                                                     {outputReshapeName});
    operators.push_back(reshapeOp1);
    tensors.push_back(new TosaSerializationTensor(outputReshapeName, reshapeDims1, inputDType0, {}));

    std::vector<int32_t> mappings {0, 1, 3, 2, 4, 5};
    std::vector<int32_t> transposeShape
    {
        static_cast<int32_t>(reshapeDims1[0]),
        static_cast<int32_t>(reshapeDims1[1]),
        static_cast<int32_t>(reshapeDims1[3]),
        static_cast<int32_t>(reshapeDims1[2]),
        static_cast<int32_t>(reshapeDims1[4]),
        static_cast<int32_t>(reshapeDims1[5])
    };
    TosaTransposeAttribute transposeAttr(mappings);
    auto* transposeOp = new TosaSerializationOperator(Op_TRANSPOSE,
                                                      Attribute_TransposeAttribute,
                                                      &transposeAttr,
                                                      {outputReshapeName},
                                                      {outputTransposeName});
    operators.push_back(transposeOp);
    tensors.push_back(new TosaSerializationTensor(outputTransposeName, transposeShape, outputDType0, {}));

    std::vector<int32_t> reshapeDims2
    {
        static_cast<int32_t>(inputs[0]->GetShape()[0]),
        static_cast<int32_t>(inputs[0]->GetShape()[1] * descriptor->m_BlockSize),
        static_cast<int32_t>(inputs[0]->GetShape()[2] * descriptor->m_BlockSize),
        static_cast<int32_t>(inputs[0]->GetShape()[3] / (descriptor->m_BlockSize * descriptor->m_BlockSize))
    };
    TosaReshapeAttribute reshapeAttr2(reshapeDims2);
    auto* reshapeOp2 = new TosaSerializationOperator(Op_RESHAPE,
                                                     Attribute_ReshapeAttribute,
                                                     &reshapeAttr2,
                                                     {outputTransposeName},
                                                     {outputName});
    operators.push_back(reshapeOp2);
    std::vector<int32_t> outputShape0 = GetTosaTensorShape(outputs[0]->GetShape());
    tensors.push_back(new TosaSerializationTensor(outputName, outputShape0, outputDType0, {}));

    return new TosaSerializationBasicBlock(blockName,     // name
                                           mainName,      // region name
                                           operators,     // operators
                                           tensors,       // tensors
                                           {inputName},   // inputs
                                           {outputName});
}