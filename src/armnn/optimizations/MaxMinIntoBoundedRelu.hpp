//
// Copyright © 2024 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "Optimization.hpp"

namespace armnn::optimizations
{

class MaxMinIntoBoundedReluImpl
{
public:
    /// Run for every exclusive connection between any Max & Min layers
    /// The Max, Min and its associated constant inputs will be removed, and replaced with a BoundedRelu Activation
    static void Run(Graph& graph, InputSlot& connection)
    {
        Layer& base = connection.GetConnectedOutputSlot()->GetOwningLayer();
        Layer& child = connection.GetOwningLayer();

        auto& maxLayer = *PolymorphicDowncast<ElementwiseBinaryLayer*>(&base);
        if (maxLayer.GetParameters().m_Operation != BinaryOperation::Maximum)
        {
            return;
        }
        auto& minLayer = *PolymorphicDowncast<ElementwiseBinaryLayer*>(&child);
        if (minLayer.GetParameters().m_Operation != BinaryOperation::Minimum)
        {
            return;
        }

        if (maxLayer.GetDataType() != minLayer.GetDataType())
        {
            return;
        }

        // get max and min values
        float_t maxValue;
        if (!GetValue(maxLayer, maxValue))
        {
            return;
        }
        float_t minValue;
        if (!GetValue(minLayer, minValue))
        {
            return;
        }

        // Save original parent output slot of the max layer
        OutputSlot& parentOut = *maxLayer.GetInputSlot(0).GetConnectedOutputSlot();

        // Insert activation layer between max layer and its parent layer
        ActivationDescriptor boundedReluDescriptor(ActivationFunction::BoundedReLu, minValue, maxValue);
        const std::string name = std::string("replaced-") + maxLayer.GetName() + std::string("-") + minLayer.GetName()
                               + std::string("-with-BoundedRelu");
        auto& boundedReluLayer = *graph.InsertNewLayer<ActivationLayer>(maxLayer.GetInputSlot(0),
                                                                        boundedReluDescriptor,
                                                                        name.c_str());

        // Reconnects with original parent.
        boundedReluLayer.GetOutputSlot().MoveAllConnections(parentOut);

        // Moves connections in min layer output to parent layer.
        // Min layer will be removed as it's left unconnected.
        // Max layer will be removed if left unconnected.
        minLayer.GetOutputSlot().MoveAllConnections(boundedReluLayer.GetOutputSlot());
    }

protected:
    MaxMinIntoBoundedReluImpl()  = default;
    ~MaxMinIntoBoundedReluImpl() = default;

private:
    static float_t GetConstTensorValue(Layer& layer)
    {
        auto& constLayer = *PolymorphicDowncast<ConstantLayer*>(&layer);
        switch (constLayer.GetDataType())
        {
            case DataType::Float32:
                return *constLayer.m_LayerOutput->GetConstTensor<float>();
            case DataType::BFloat16:
                return static_cast<float_t>(*constLayer.m_LayerOutput->GetConstTensor<BFloat16>());
            case DataType::Float16:
                return static_cast<float_t>(*constLayer.m_LayerOutput->GetConstTensor<half_float::half>());
            case DataType::QAsymmU8:
            case DataType::Boolean:
                return static_cast<float_t>(*constLayer.m_LayerOutput->GetConstTensor<uint8_t>());
            case DataType::QAsymmS8:
            case DataType::QSymmS8:
                return static_cast<float_t>(*constLayer.m_LayerOutput->GetConstTensor<int8_t>());
            case DataType::QSymmS16:
                return static_cast<float_t>(*constLayer.m_LayerOutput->GetConstTensor<int16_t>());
            case DataType::Signed32:
                return static_cast<float_t>(*constLayer.m_LayerOutput->GetConstTensor<int32_t>());
            case DataType::Signed64:
                return static_cast<float_t>(*constLayer.m_LayerOutput->GetConstTensor<int64_t>());
            default:
                throw InvalidArgumentException("No supported Data Type");
        }
    }

    static bool GetValue(Layer& layer, float_t& value)
    {
        Layer& input0 = layer.GetInputSlot(0).GetConnectedOutputSlot()->GetOwningLayer();
        Layer& input1 = layer.GetInputSlot(1).GetConnectedOutputSlot()->GetOwningLayer();
        if (input0.GetType() == LayerType::Constant)
        {
            if (input0.GetOutputSlot(0).GetTensorInfo().GetNumElements() != 1)
            {
                return false;
            }
            value = GetConstTensorValue(input0);
        }
        else if (input1.GetType() == LayerType::Constant)
        {
            if (input1.GetOutputSlot(0).GetTensorInfo().GetNumElements() != 1)
            {
                return false;
            }
            value = GetConstTensorValue(input1);
        }
        else
        {
            return false;
        }
        return true;
    };
};

using MaxMinIntoBoundedRelu = OptimizeForExclusiveConnection<ElementwiseBinaryLayer,
                                                             ElementwiseBinaryLayer,
                                                             MaxMinIntoBoundedReluImpl>;

} // namespace armnn::optimizations