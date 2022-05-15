//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include "Optimization.hpp"
#include "NetworkUtils.hpp"

namespace armnn
{
namespace optimizations
{

class ConvertConstDequantisationLayersToConstLayersImpl
{
public:
    void Run(Graph& graph, InputSlot& connection) const
    {
        Layer& base = connection.GetConnectedOutputSlot()->GetOwningLayer();
        Layer& child = connection.GetOwningLayer();

        ARMNN_ASSERT(base.GetType() == LayerType::Constant);
        ARMNN_ASSERT(child.GetType() == LayerType::Dequantize);

        ReplaceConstDequantisationLayer(graph,
                                        PolymorphicDowncast<ConstantLayer*>(&base),
                                        PolymorphicDowncast<DequantizeLayer*>(&child));

    }
protected:
    ConvertConstDequantisationLayersToConstLayersImpl() = default;
    ~ConvertConstDequantisationLayersToConstLayersImpl() = default;
private:

    static void ReplaceConstDequantisationLayer(Graph& graph,
                                                ConstantLayer* constantLayer,
                                                DequantizeLayer* dequantizeLayer)
    {
        IgnoreUnused(graph);
        /**
         * This optimisation is to find situations where a constant set of inputs is being provided to a Dequantization
         * layer. In this case we don't want the overhead of Dequantizing the values on every inference, instead we
         * want to Dequantize them once and store them in a Const layer to be used everytime as they will not change.
         */
        TensorInfo constantInfo = constantLayer->GetOutputSlot(0).GetTensorInfo();
        TensorInfo inputDequantizeInfo = dequantizeLayer->GetInputSlot(0).GetConnectedOutputSlot()->GetTensorInfo();
        TensorInfo outputDequantizeInfo = dequantizeLayer->GetOutputSlot(0).GetTensorInfo();

        ARMNN_ASSERT(constantLayer->GetNumOutputSlots() == 1);
        auto numConnections = constantLayer->GetOutputSlot(0).GetNumConnections();

        std::vector<float> newValues(outputDequantizeInfo.GetNumElements());
        if (constantInfo.GetDataType() == DataType::Float16 &&
            inputDequantizeInfo.GetDataType() == DataType::Float16 &&
            outputDequantizeInfo.GetDataType() == DataType::Float32)
        {
            armnnUtils::FloatingPointConverter::ConvertFloat16To32(constantLayer->m_LayerOutput->Map(true),
                                                                   outputDequantizeInfo.GetNumElements(),
                                                                   newValues.data());
        }
        else if (constantInfo.GetDataType() == DataType::QAsymmS8 &&
                inputDequantizeInfo.GetDataType() == DataType::QAsymmS8 &&
                outputDequantizeInfo.GetDataType() == DataType::Float32)
        {
            ConvertInt8To32(constantLayer->m_LayerOutput->Map(true),
                            outputDequantizeInfo.GetNumElements(),
                            newValues.data());
        }

        TensorInfo newInfo = outputDequantizeInfo;
        newInfo.SetConstant(true);
        ConstTensor newInput(newInfo, newValues);
        constantLayer->m_LayerOutput.reset(new ScopedTensorHandle(newInput));

        // Moves connections in dequantize output to the constant layer.
        // Dequantize layer will be removed if left unconnected.
        dequantizeLayer->GetOutputSlot().MoveAllConnections(constantLayer->GetOutputSlot());

        // Updating the output tensor
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


static void ConvertInt8To32(const void* srcInt8Buffer,
                            size_t numElements,
                            float* dstFloat32Buffer)
{
    ARMNN_ASSERT(srcInt8Buffer != nullptr);
    ARMNN_ASSERT(dstFloat32Buffer != nullptr);

    const auto* pInt8 = static_cast<const int8_t*>(srcInt8Buffer);

    for (size_t i = 0; i < numElements; ++i)
    {
        dstFloat32Buffer[i] = pInt8[i];
    }
}

};

using ConvertConstDequantisationLayersToConstLayers
    = OptimizeForConnection<ConstantLayer,
                            DequantizeLayer,
                            ConvertConstDequantisationLayersToConstLayersImpl>;

} // namespace optimizations
} // namespace armnn