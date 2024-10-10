//
// Copyright © 2024 Arm Ltd and Contributors.
// SPDX-License-Identifier: MIT
//
#pragma once

#include "Optimization.hpp"
#include "NetworkUtils.hpp"

#include <armnn/Logging.hpp>
#include <armnnUtils/Permute.hpp>

namespace armnn
{
namespace optimizations
{

class TurboConvertConstDequantisationLayersToConstLayersImpl
{
public:
    void Run(Graph& graph, InputSlot& connection) const
    {
        Layer& base = connection.GetConnectedOutputSlot()->GetOwningLayer();
        Layer& child = connection.GetOwningLayer();

        // Check the basic criteria for the optimization are met.
        if ((base.GetType() == LayerType::Constant) && (child.GetType() == LayerType::Dequantize))
        {
            ReplaceConstDequantisationLayer(graph, PolymorphicDowncast<ConstantLayer*>(&base),
                                            PolymorphicDowncast<DequantizeLayer*>(&child));
        }
    }
protected:
    TurboConvertConstDequantisationLayersToConstLayersImpl() = default;
    ~TurboConvertConstDequantisationLayersToConstLayersImpl() = default;
private:

    static void ReplaceConstDequantisationLayer(Graph&,
                                                ConstantLayer* constantLayer,
                                                DequantizeLayer* dequantizeLayer)
    {
        ARMNN_LOG(info) << "TurboConvertConstDequantisationLayersToConstLayersImpl::ReplaceConstDequantisationLayer()";
        /**
         * This optimisation is to find situations where a constant set of inputs is being provided to a Dequantization
         * layer. In this case we don't want the overhead of Dequantizing the values on every inference, instead we
         * want to Dequantize them once and store them in a Const layer to be used everytime as they will not change.
         */
        TensorInfo constantInfo = constantLayer->GetOutputSlot(0).GetTensorInfo();
        TensorInfo inputDequantizeInfo = dequantizeLayer->GetInputSlot(0).GetConnectedOutputSlot()->GetTensorInfo();
        TensorInfo outputDequantizeInfo = dequantizeLayer->GetOutputSlot(0).GetTensorInfo();

        bool requiresPermute = false;

        auto connection = dequantizeLayer->GetOutputSlot(0).GetConnection(0);
        if (connection)
        {
            if (connection->GetOwningLayer().GetType() == LayerType::Convolution2d)
            {
                /**
                 * ArmNN does not currently support non-fixed weights or bias
                 * The NNAPI filter is always OHWI [depth_out, filter_height, filter_width, depth_in]
                 * but ArmNN expects the filter's height and width indices to match the input's height
                 * and width indices so we permute it to OIHW if the DataLayout is NCHW
                 */
                ARMNN_LOG(info) << "ConvertConstDequantisationLayersToConstLayersImpl:: Connected to "
                                   "Convolution layer.";
                auto conv2dLayer = PolymorphicDowncast<Convolution2dLayer*>(&connection->GetOwningLayer());
                if (conv2dLayer->GetParameters().m_DataLayout == DataLayout::NCHW)
                {
                    ARMNN_LOG(info) << "ConvertConstDequantisationLayersToConstLayersImpl:: Connected to "
                                        "Convolution layer and requires permute on weights. ";
                    requiresPermute = true;
                }
            }
        }

        auto numConnections = constantLayer->GetOutputSlot(0).GetNumConnections();

        ARMNN_LOG(info) << "constantInfo datatype:" << armnn::GetDataTypeName(constantInfo.GetDataType())
           << "inputDequantizeInfo datatype:" << armnn::GetDataTypeName(inputDequantizeInfo.GetDataType())
           << "outputDequantizeInfo datatype:" << armnn::GetDataTypeName(outputDequantizeInfo.GetDataType());

        TensorInfo newInfo = inputDequantizeInfo;
        newInfo.SetConstant(true);
        if (requiresPermute)
        {
            ARMNN_LOG(info) << "TurboConvertConstDequantisationLayersToConstLayersImpl:: Permuting the constant data.";
            const PermutationVector OHWIToOIHW = {0, 2, 3, 1};
            // Here Permute weights
            std::vector<Half> permutedValues(outputDequantizeInfo.GetNumElements());
            armnnUtils::Permute(outputDequantizeInfo.GetShape(), OHWIToOIHW,
                                constantLayer->m_LayerOutput->Map(true), permutedValues.data(),
                                GetDataTypeSize(outputDequantizeInfo.GetDataType()));
            ConstTensor newInput(newInfo, permutedValues);
            constantLayer->m_LayerOutput.reset(new ScopedTensorHandle(newInput));
        }
        else
        {
            ConstTensor newInput(newInfo, constantLayer->m_LayerOutput->Map(true));
            constantLayer->m_LayerOutput.reset(new ScopedTensorHandle(newInput));
        }

        // Move connections in dequantize output to the constant layer.
        // Dequantize layer will be removed if left unconnected.
        dequantizeLayer->GetOutputSlot().MoveAllConnections(constantLayer->GetOutputSlot());

        // Update the output tensor
        constantLayer->GetOutputSlot(0).SetTensorInfo(newInfo);
        ARMNN_ASSERT(constantLayer->GetOutputSlot(0).GetTensorInfo().IsConstant() == true);

        // Set isConstant to true in all input tensor infos where constantLayer is now connected to
        for (unsigned int i = numConnections; i < constantLayer->GetOutputSlot(0).GetNumConnections(); ++i)
        {
            auto info = constantLayer->GetOutputSlot(0).GetConnection(i)->GetOwningLayer().GetInputSlot(0)
                    .GetConnectedOutputSlot()->GetTensorInfo();
            info.SetConstant();
            constantLayer->GetOutputSlot(0).GetConnection(i)->GetOwningLayer().GetInputSlot(0)
                    .GetConnectedOutputSlot()->SetTensorInfo(info);
        }
    }

};

using TurboConvertConstDequantisationLayersToConstLayers =
                            OptimizeForConnection<ConstantLayer,
                            DequantizeLayer,
                            TurboConvertConstDequantisationLayersToConstLayersImpl>;

} // namespace optimizations
} // namespace armnn