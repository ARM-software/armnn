//
// Copyright © 2022-2025 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "TosaMappings.hpp"
#include "tosaCommon/operatorMappings/PReluOperator.hpp"

using namespace armnn;
using namespace tosa;

TosaSerializationBasicBlock* CreateEmptyTosaSerializationBasicBlock()
{
    // Empty basic block when no TOSA mapping implemented/exists
    auto* op = new TosaSerializationOperator(Op_UNKNOWN, Attribute_NONE, nullptr, {}, {});
    return new TosaSerializationBasicBlock("", "", {op}, {}, {}, {});
}

TosaSerializationBasicBlock* GetTosaMapping(const Layer* layer,
                                            const LayerType type,
                                            const std::vector<const TensorInfo*>& inputs,
                                            const std::vector<const TensorInfo*>& outputs,
                                            const BaseDescriptor& descriptor)
{
    switch (type)
    {
        case LayerType::Activation:
        {
            auto activationDesc = PolymorphicDowncast<const ActivationDescriptor*>(&descriptor);
            switch (activationDesc->m_Function)
            {
                case ActivationFunction::LeakyReLu:
                {
                    return ConvertLeakyReluToTosaOperator(layer, inputs, outputs, activationDesc);
                }
                case ActivationFunction::ReLu:
                case ActivationFunction::BoundedReLu:
                {
                    return ConvertReluToTosaOperator(layer, inputs, outputs, activationDesc);
                }
                case ActivationFunction::Gelu:
                {
                    return ConvertGeluToTosaOperator(layer, inputs, outputs, activationDesc);
                }
                case ActivationFunction::HardSwish:
                {
                    return ConvertHardSwishToTosaOperator(layer, inputs, outputs, activationDesc);
                }
                case ActivationFunction::Sigmoid:
                {
                    return ConvertSigmoidToTosaOperator(layer, inputs, outputs, activationDesc);
                }
                case ActivationFunction::TanH:
                {
                    return ConvertTanHToTosaOperator(layer, inputs, outputs, activationDesc);
                }
                default:
                {
                    return CreateEmptyTosaSerializationBasicBlock();
                }
            }
        }
        case LayerType::Addition:
        case LayerType::Multiplication:
        case LayerType::Subtraction:
        {
            return ConvertElementwiseBinaryToTosaOperator(layer, type, inputs, outputs);
        }
        case LayerType::ElementwiseBinary:
        {
            auto binaryDesc = PolymorphicDowncast<const ElementwiseBinaryDescriptor*>(&descriptor);
            switch (binaryDesc->m_Operation)
            {
                case BinaryOperation::SqDiff:
                    return ConvertSquaredDifferenceToTosaOperator(layer, type, inputs, outputs, binaryDesc);
                default:
                    return ConvertElementwiseBinaryToTosaOperator(layer, type, inputs, outputs, binaryDesc);
            }
        }
        case LayerType::ElementwiseUnary:
        {
            auto unaryDesc = PolymorphicDowncast<const ElementwiseUnaryDescriptor*>(&descriptor);
            switch(unaryDesc->m_Operation)
            {
                case UnaryOperation::Rsqrt:
                {
                    return ConvertRsqrtOperator(layer, inputs, outputs, unaryDesc);
                }
                case UnaryOperation::Exp:
                {
                    return ConvertExpOperator(layer, inputs, outputs, unaryDesc);
                }
                case UnaryOperation::Log:
                {
                    return ConvertLogOperator(layer, inputs, outputs, unaryDesc);
                }
                default:
                {
                    return CreateEmptyTosaSerializationBasicBlock();
                }
            }
        }
        case LayerType::BatchMatMul:
        {
            auto batchMatMulDesc = PolymorphicDowncast<const BatchMatMulDescriptor*>(&descriptor);
            return ConvertBatchMatMulToTosaOperator(layer, inputs, outputs, batchMatMulDesc);
        }
        case LayerType::Concat:
        {
            auto concatDesc = PolymorphicDowncast<const OriginsDescriptor*>(&descriptor);
            return ConvertConcatToTosaOperator(layer, inputs, outputs, concatDesc);
        }
        case LayerType::Constant:
        {
            bool isDepthwiseConv2dWeights = false;
            if(layer)
            {
                // The difference in layout of weights in Tensorflow/ArmNN and the layout
                // described in TOSA means we must permute the weights from [1, H, W, C * M] to [H, W, C, M].
                unsigned int slotIdx = layer->GetOutputSlot().GetConnection(0)->GetSlotIndex();
                LayerType type = layer->GetOutputSlot().GetConnection(0)->GetOwningLayer().GetType();
                if(type == LayerType::DepthwiseConvolution2d && slotIdx == 1)
                {
                    isDepthwiseConv2dWeights = true;
                }
            }
            return ConvertConstantToTosaOperator(layer, outputs, isDepthwiseConv2dWeights);
        }
        case LayerType::Convolution2d:
        {
            auto conv2dDesc = PolymorphicDowncast<const Convolution2dDescriptor*>(&descriptor);
            return ConvertConv2dToTosaOperator(layer, inputs, outputs, conv2dDesc);
        }
        case LayerType::Convolution3d:
        {
            auto conv3dDesc = PolymorphicDowncast<const Convolution3dDescriptor*>(&descriptor);
            return ConvertConv3dToTosaOperator(layer, inputs, outputs, conv3dDesc);
        }
        case LayerType::DepthwiseConvolution2d:
        {
            auto conv2dDesc = PolymorphicDowncast<const DepthwiseConvolution2dDescriptor*>(&descriptor);
            return ConvertDepthwiseConv2dToTosaOperator(layer, inputs, outputs, conv2dDesc);
        }
        case LayerType::DepthToSpace:
        {
            auto desc = PolymorphicDowncast<const DepthToSpaceDescriptor*>(&descriptor);
            return ConvertDepthToSpaceToTosaOperator(layer, inputs, outputs, desc);
        }
        case LayerType::FullyConnected:
        {
            auto fullyConnectedDesc = PolymorphicDowncast<const FullyConnectedDescriptor*>(&descriptor);
            return ConvertFullyConnectedToTosaOperator(layer, inputs, outputs, fullyConnectedDesc);
        }
        case LayerType::Gather:
        {
            auto gatherDesc = PolymorphicDowncast<const GatherDescriptor*>(&descriptor);
            return ConvertGatherToTosaOperator(layer, inputs, outputs, gatherDesc);
        }
        case LayerType::Pad:
        {
            auto padDesc = PolymorphicDowncast<const PadDescriptor*>(&descriptor);
            return ConvertPadToTosaOperator(layer, inputs, outputs, padDesc);
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
        case LayerType::Mean:
        {
            auto meanDesc = PolymorphicDowncast<const MeanDescriptor*>(&descriptor);

            ReduceDescriptor reduceDesc;
            reduceDesc.m_KeepDims        = meanDesc->m_KeepDims;
            reduceDesc.m_vAxis           = meanDesc->m_Axis;
            reduceDesc.m_ReduceOperation = ReduceOperation::Mean;

            return ConvertReduceToTosaOperator(layer, inputs, outputs, &reduceDesc);
        }
        case LayerType::Dequantize:
        {
            return ConvertDequantizeToTosaOperator(layer, inputs, outputs);
        }
        case LayerType::Quantize:
        {
            return ConvertQuantizeToTosaOperator(layer, inputs, outputs);
        }
        case LayerType::Prelu:
        {
            return ConvertPReluToTosaOperator(layer, inputs, outputs);
        }
        case LayerType::Reduce:
        {
            auto reduceDesc = PolymorphicDowncast<const ReduceDescriptor*>(&descriptor);
            return ConvertReduceToTosaOperator(layer, inputs, outputs, reduceDesc);
        }
        case LayerType::Reshape:
        {
            auto reshapeDesc = PolymorphicDowncast<const ReshapeDescriptor*>(&descriptor);
            return ConvertReshapeToTosaOperator(layer, inputs, outputs, reshapeDesc);
        }
        case LayerType::Resize:
        {
            auto resizeDesc = PolymorphicDowncast<const ResizeDescriptor*>(&descriptor);
            return ConvertResizeToTosaOperator(layer, inputs, outputs, resizeDesc);
        }
        case LayerType::Slice:
        {
            auto sliceDesc = PolymorphicDowncast<const SliceDescriptor*>(&descriptor);
            return ConvertSliceToTosaOperator(layer, inputs, outputs, sliceDesc);
        }
        case LayerType::Softmax:
        {
            auto softmaxDesc = PolymorphicDowncast<const SoftmaxDescriptor*>(&descriptor);
            return ConvertSoftmaxToTosaOperator(layer, inputs, outputs, softmaxDesc);
        }
        case LayerType::Splitter:
        {
            auto splitDesc = PolymorphicDowncast<const SplitterDescriptor*>(&descriptor);
            return ConvertSplitToTosaOperator(layer, inputs, outputs, splitDesc);
        }
        case LayerType::Stack:
        {
            auto stackDesc = PolymorphicDowncast<const StackDescriptor*>(&descriptor);
            return ConvertStackToTosaOperator(layer, inputs, outputs, stackDesc);
        }
        case LayerType::StridedSlice:
        {
            auto sliceDesc = PolymorphicDowncast<const StridedSliceDescriptor*>(&descriptor);
            return ConvertStridedSliceToTosaOperator(layer, inputs, outputs, sliceDesc);
        }
        case LayerType::TransposeConvolution2d:
        {
            auto transposeConv2dDesc = PolymorphicDowncast<const TransposeConvolution2dDescriptor*>(&descriptor);
            return ConvertTransposeConv2dToTosaOperator(layer, inputs, outputs, transposeConv2dDesc);
        }
        case LayerType::Transpose:
        {
            auto transposeDesc = PolymorphicDowncast<const TransposeDescriptor*>(&descriptor);
            return ConvertTransposeToTosaOperator(layer, inputs, outputs, transposeDesc);
        }
        default:
        {
            return CreateEmptyTosaSerializationBasicBlock();
        }
    }
}

TosaSerializationBasicBlock* GetTosaMappingFromLayer(const Layer* layer)
{
    std::vector<const TensorInfo*> inputs;
    for (auto const& inputSlot : layer->GetInputSlots())
    {
        inputs.emplace_back(&inputSlot.GetTensorInfo());
    }

    std::vector<const TensorInfo*> outputs;
    for (auto& outputSlot : layer->GetOutputSlots())
    {
        outputs.emplace_back(&outputSlot.GetTensorInfo());
    }

    TosaSerializationBasicBlock* basicBlock = GetTosaMapping(layer,
                                                             layer->GetType(),
                                                             inputs,
                                                             outputs,
                                                             layer->GetParameters());

    return basicBlock;
}
