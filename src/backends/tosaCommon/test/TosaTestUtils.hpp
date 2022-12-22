//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <Layer.hpp>

#include <tosaCommon/TosaMappings.hpp>
#include <tosaCommon/operatorMappings/TosaOperatorUtils.hpp>

#include <doctest/doctest.h>
#include <numeric>

using namespace armnn;
using namespace tosa;

inline void VerifyTosaAttribute(const BaseDescriptor& descriptor,
                                const TosaAttributeBase* attribute,
                                std::vector<int32_t> inputShape,
                                std::vector<int32_t> outputShape,
                                LayerType type,
                                uint32_t mappingOpNumber = 0)
{
    switch (type)
    {
        case LayerType::Convolution2d:
        {
            auto conv2dDesc = PolymorphicDowncast<const Convolution2dDescriptor*>(&descriptor);
            std::vector<int> pad = {static_cast<int>(conv2dDesc->m_PadTop),
                                    static_cast<int>(conv2dDesc->m_PadBottom),
                                    static_cast<int>(conv2dDesc->m_PadLeft),
                                    static_cast<int>(conv2dDesc->m_PadRight)};

            std::vector<int> dilation = {static_cast<int>(conv2dDesc->m_DilationY),
                                         static_cast<int>(conv2dDesc->m_DilationX)};
            std::vector<int> stride = {static_cast<int>(conv2dDesc->m_StrideY),
                                       static_cast<int>(conv2dDesc->m_StrideX)};
            TosaConvAttribute convAttribute(attribute);
            CHECK(pad == convAttribute.pad());
            CHECK(dilation == convAttribute.dilation());
            CHECK(stride == convAttribute.stride());
            break;
        }
        case LayerType::Pooling2d:
        {
            auto poolDesc = PolymorphicDowncast<const Pooling2dDescriptor*>(&descriptor);
            std::vector<int> pad = {static_cast<int>(poolDesc->m_PadTop),
                                    static_cast<int>(poolDesc->m_PadBottom),
                                    static_cast<int>(poolDesc->m_PadLeft),
                                    static_cast<int>(poolDesc->m_PadRight)};

            bool avgPoolIgnoreValue =
                     (poolDesc->m_PoolType == PoolingAlgorithm::Average) &&
                     (poolDesc->m_PaddingMethod == PaddingMethod::IgnoreValue);
            if (avgPoolIgnoreValue)
            {
                if (mappingOpNumber == 0)
                {
                    if (poolDesc->m_DataLayout == DataLayout::NHWC)
                    {
                        pad = {0,
                               0,
                               static_cast<int>(poolDesc->m_PadTop),
                               static_cast<int>(poolDesc->m_PadBottom),
                               static_cast<int>(poolDesc->m_PadLeft),
                               static_cast<int>(poolDesc->m_PadRight),
                               0,
                               0
                        };
                    }
                    else
                    {
                        pad = {0,
                               0,
                               0,
                               0,
                               static_cast<int>(poolDesc->m_PadTop),
                               static_cast<int>(poolDesc->m_PadBottom),
                               static_cast<int>(poolDesc->m_PadLeft),
                               static_cast<int>(poolDesc->m_PadRight)
                        };
                    }

                    TosaPadAttribute padAttribute(attribute);

                    CHECK(pad == padAttribute.padding());
                    CHECK(0.0f == padAttribute.pad_const_fp());
                    CHECK(0 == padAttribute.pad_const_int());

                    break;
                }
                pad = {0, 0, 0, 0};
            }

            std::vector<int> kernel = {static_cast<int>(poolDesc->m_PoolHeight),
                                       static_cast<int>(poolDesc->m_PoolWidth)};
            std::vector<int> stride = {static_cast<int>(poolDesc->m_StrideY),
                                       static_cast<int>(poolDesc->m_StrideX)};
            TosaPoolAttribute poolAttribute(attribute);
            CHECK(pad == poolAttribute.pad());
            CHECK(kernel == poolAttribute.kernel());
            CHECK(stride == poolAttribute.stride());
            break;
        }
        case LayerType::Reshape:
        {
            auto reshapeDesc = PolymorphicDowncast<const ReshapeDescriptor*>(&descriptor);
            TosaReshapeAttribute reshapeAttribute(attribute);
            std::vector<int32_t> shapeAttrib = reshapeAttribute.new_shape();

            CHECK(GetTosaTensorShape(reshapeDesc->m_TargetShape) == shapeAttrib);
            CHECK(outputShape == shapeAttrib);

            auto numInputElements = std::accumulate(std::begin(inputShape),
                                                    std::end(inputShape),
                                                    1,
                                                    std::multiplies<int32_t>());
            auto numAttributeShapeElements = std::accumulate(std::begin(shapeAttrib),
                                                             std::end(shapeAttrib),
                                                             1,
                                                             std::multiplies<int32_t>());
            CHECK(numInputElements == numAttributeShapeElements);

            break;
        }
        case LayerType::Slice:
        {
            auto sliceDesc = PolymorphicDowncast<const SliceDescriptor*>(&descriptor);
            TosaSliceAttribute reshapeAttribute(attribute);

            std::vector<int32_t> begin(sliceDesc->m_Begin.begin(), sliceDesc->m_Begin.end());
            std::vector<int32_t> size(sliceDesc->m_Size.begin(), sliceDesc->m_Size.end());

            CHECK(begin == reshapeAttribute.start());
            CHECK(size == reshapeAttribute.size());

            CHECK(begin.size() == inputShape.size());
            CHECK(size.size() == inputShape.size());

            CHECK(begin.size() == outputShape.size());
            CHECK(size.size() == outputShape.size());

            break;
        }
        case LayerType::TransposeConvolution2d:
        {
            auto transposeConv2dDesc = PolymorphicDowncast<const TransposeConvolution2dDescriptor*>(&descriptor);
            std::vector<int> outPad = {-static_cast<int>(transposeConv2dDesc->m_PadTop),
                                       -static_cast<int>(transposeConv2dDesc->m_PadBottom),
                                       -static_cast<int>(transposeConv2dDesc->m_PadLeft),
                                       -static_cast<int>(transposeConv2dDesc->m_PadRight)};
            std::vector<int> stride = {static_cast<int>(transposeConv2dDesc->m_StrideY),
                                       static_cast<int>(transposeConv2dDesc->m_StrideX)};
            TosaTransposeConvAttribute transposeConvAttribute(attribute);
            CHECK(outPad == transposeConvAttribute.out_pad());
            CHECK(stride == transposeConvAttribute.stride());
            break;
        }
        case LayerType::Transpose:
        {
            auto transposeDesc = PolymorphicDowncast<const TransposeDescriptor*>(&descriptor);
            std::vector<int> outPerm(transposeDesc->m_DimMappings.begin(), transposeDesc->m_DimMappings.end());
            TosaTransposeAttribute transposeAttribute(attribute);
            CHECK(outPerm == transposeAttribute.perms());
            break;
        }
        default:
            break;
    }
    return;
}

inline void AssertTosaOneToOneMappingBasicBlock(TosaSerializationBasicBlock* basicBlock,
                                                std::vector<std::vector<int32_t>> inputShape,
                                                std::vector<std::vector<int32_t>> outputShape,
                                                Op tosaOp,
                                                Attribute tosaAttribute,
                                                const BaseDescriptor& descriptor,
                                                LayerType type,
                                                DType dataType = DType_FP32)
{
    uint32_t numInputs = static_cast<uint32_t>(inputShape.size());
    uint32_t numInputTensors = static_cast<uint32_t>(inputShape.size());
    uint32_t numOutputs = static_cast<uint32_t>(outputShape.size());
    std::string operatorString = TosaOpToString(tosaOp);

    // The number of tensors in the block can be different if there are constant layers, as they are created separately.
    if(type == LayerType::Convolution2d)
    {
        numInputTensors = PolymorphicDowncast<const Convolution2dDescriptor*>(&descriptor)->m_BiasEnabled ? 3 : 2;
    }

    std::string blockStr = operatorString + "_block_";
    CHECK(basicBlock->GetName().find(blockStr)  != std::string::npos);
    CHECK(basicBlock->GetInputs().size() == numInputTensors);
    CHECK(basicBlock->GetOutputs().size() == numOutputs);
    CHECK(basicBlock->GetOperators().size() == 1);
    CHECK(basicBlock->GetTensors().size() == (numInputs + numOutputs));

    TosaSerializationOperator* op = basicBlock->GetOperators().at(0);
    CHECK(op->GetInputTensorNames().size() == numInputTensors);
    CHECK(op->GetOutputTensorNames().size() == numOutputs);

    for (uint32_t i = 0; i < numInputs; i++)
    {
        std::basic_string<char> blockInputName = basicBlock->GetInputs()[i];
        std::basic_string<char> operatorInputName  = op->GetInputTensorNames()[i];
        std::basic_string<char> tensorName = basicBlock->GetTensors()[i]->GetName();

        std::string opStr = "input" + std::to_string(i) + "_";

        CHECK(blockInputName == operatorInputName);
        CHECK(tensorName == operatorInputName);
        CHECK(blockInputName.find(opStr) != std::string::npos);
    }

    for (uint32_t i = 0; i < numOutputs; i++)
    {
        std::basic_string<char> blockOutputName = basicBlock->GetOutputs()[i];
        std::basic_string<char> operatorOutputName  = op->GetOutputTensorNames()[i];
        std::basic_string<char> tensorName = basicBlock->GetTensors()[numInputs + i]->GetName();

        std::string opStr = "output" + std::to_string(i) + "_";
        if (tosaOp == Op_CONST)
        {
            opStr = "constant_";
        }

        CHECK(blockOutputName == operatorOutputName);
        CHECK(tensorName == operatorOutputName);
        CHECK(blockOutputName.find(opStr)  != std::string::npos);
    }

    CHECK(op->GetAttributeType() == tosaAttribute);
    CHECK(op->GetOp() == tosaOp);

    for (uint32_t i = 0; i < numInputs; i++)
    {
        TosaSerializationTensor* tensor = basicBlock->GetTensors()[i];
        CHECK(tensor->GetDtype() == dataType);
        CHECK(tensor->GetData().size() == 0);
        CHECK(tensor->GetShape() == inputShape[static_cast<unsigned long int>(i)]);
    }

    for (uint32_t i = 0; i < numOutputs; i++)
    {
        TosaSerializationTensor* tensor = basicBlock->GetTensors()[i + inputShape.size()];
        CHECK(tensor->GetDtype() == dataType);
        CHECK(tensor->GetShape() == outputShape[static_cast<unsigned long int>(i)]);
        if (tosaOp != Op_CONST)
        {
            // Const tensors contain data.
            CHECK(tensor->GetData().size() == 0);
        }
    }

    std::vector<int32_t> input = {};
    std::vector<int32_t> output = {};

    if (!inputShape.empty())
    {
        input = inputShape[0];
    }

    if (!outputShape.empty())
    {
        output = outputShape[0];
    }

    VerifyTosaAttribute(descriptor,
                        op->GetAttribute(),
                        input,
                        output,
                        type);
}