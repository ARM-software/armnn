//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <Layer.hpp>

#include <tosaCommon/TosaMappings.hpp>

#include <doctest/doctest.h>

using namespace armnn;
using namespace tosa;

inline void VerifyTosaAttributeFromDescriptor(const BaseDescriptor& descriptor,
                                              const TosaAttributeBase* attribute,
                                              LayerType type,
                                              uint32_t mappingOpNumber = 0)
{
    switch (type)
    {
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
    uint32_t numOutputs = static_cast<uint32_t>(outputShape.size());
    std::string operatorString = TosaOpToString(tosaOp);

    std::string blockStr = operatorString + "_block_";
    CHECK(basicBlock->GetName().find(blockStr)  != std::string::npos);
    CHECK(basicBlock->GetInputs().size() == numInputs);
    CHECK(basicBlock->GetOutputs().size() == numOutputs);
    CHECK(basicBlock->GetOperators().size() == 1);
    CHECK(basicBlock->GetTensors().size() == (numInputs + numOutputs));

    TosaSerializationOperator* op = basicBlock->GetOperators().at(0);
    CHECK(op->GetInputTensorNames().size() == numInputs);
    CHECK(op->GetOutputTensorNames().size() == numOutputs);

    for (uint32_t i = 0; i < numInputs; i++)
    {
        std::basic_string<char> blockInputName = basicBlock->GetInputs()[i];
        std::basic_string<char> operatorInputName  = op->GetInputTensorNames()[i];
        std::basic_string<char> tensorName = basicBlock->GetTensors()[i]->GetName();

        std::string opStr = operatorString + "_input" + std::to_string(i) + "_";

        CHECK(blockInputName == operatorInputName);
        CHECK(tensorName == operatorInputName);
        CHECK(blockInputName.find(opStr)  != std::string::npos);
    }

    for (uint32_t i = 0; i < numOutputs; i++)
    {
        std::basic_string<char> blockOutputName = basicBlock->GetOutputs()[i];
        std::basic_string<char> operatorOutputName  = op->GetOutputTensorNames()[i];
        std::basic_string<char> tensorName = basicBlock->GetTensors()[numInputs + i]->GetName();

        std::string opStr = operatorString + "_output" + std::to_string(i) + "_";

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
        CHECK(tensor->GetData().size() == 0);
        CHECK(tensor->GetShape() == outputShape[static_cast<unsigned long int>(i)]);
    }

    VerifyTosaAttributeFromDescriptor(descriptor,
                                      op->GetAttribute(),
                                      type);
}