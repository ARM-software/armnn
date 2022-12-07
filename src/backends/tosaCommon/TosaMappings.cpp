//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "TosaMappings.hpp"

using namespace armnn;
using namespace tosa;

TosaSerializationBasicBlock* CreateEmptyTosaSerializationBasicBlock()
{
    // Empty basic block when no TOSA mapping implemented/exists
    auto* op = new TosaSerializationOperator(Op_UNKNOWN, Attribute_NONE, nullptr, {}, {});
    return new TosaSerializationBasicBlock("", {op}, {}, {}, {});
}

TosaSerializationBasicBlock* GetTosaMapping(const Layer* layer,
                                            const LayerType type,
                                            const std::vector<const TensorInfo*>& inputs,
                                            const std::vector<const TensorInfo*>& outputs,
                                            const BaseDescriptor& descriptor)
{
    switch (type)
    {
        case LayerType::Addition:
        {
            return ConvertAdditionToTosaOperator(layer, inputs, outputs);
        }
        case LayerType::Constant:
        {
            return ConvertConstantToTosaOperator(layer, outputs);
        }
        case LayerType::Convolution2d:
        {
            auto conv2dDesc = PolymorphicDowncast<const Convolution2dDescriptor*>(&descriptor);
            return ConvertConv2dToTosaOperator(layer, inputs, outputs, conv2dDesc);
        }
        case LayerType::Pooling2d:
        {
            auto poolDesc = PolymorphicDowncast<const Pooling2dDescriptor*>(&descriptor);

            bool avgPoolIgnoreValue =
                (poolDesc->m_PoolType == PoolingAlgorithm::Average) &&
                (poolDesc->m_PaddingMethod == PaddingMethod::IgnoreValue);

            if (poolDesc->m_PoolType == PoolingAlgorithm::L2)
            {
                return CreateEmptyTosaSerializationBasicBlock();
            }
            else if (avgPoolIgnoreValue)
            {
                return ConvertAvgPool2DIgnoreValueToTosaOperator(layer, inputs, outputs, poolDesc);
            }
            else
            {
                return ConvertPooling2DToTosaOperator(layer, inputs, outputs, poolDesc);
            }
        }
        case LayerType::Reshape:
        {
            auto reshapeDesc = PolymorphicDowncast<const ReshapeDescriptor*>(&descriptor);
            return ConvertReshapeToTosaOperator(layer, inputs, outputs, reshapeDesc);
        }
        default:
        {
            return CreateEmptyTosaSerializationBasicBlock();
        }
    }
}

TosaSerializationBasicBlock* GetTosaMappingFromLayer(Layer* layer)
{
    std::vector<const TensorInfo*> inputs;
    for (auto inputSlot : layer->GetInputSlots())
    {
        inputs.push_back(&inputSlot.GetConnection()->GetTensorInfo());
    }

    std::vector<const TensorInfo*> outputs;
    for (auto& outputSlot : layer->GetOutputSlots())
    {
        outputs.push_back(&outputSlot.GetTensorInfo());
    }

    TosaSerializationBasicBlock* basicBlock = GetTosaMapping(layer,
                                                             layer->GetType(),
                                                             inputs,
                                                             outputs,
                                                             layer->GetParameters());
    return basicBlock;
}
