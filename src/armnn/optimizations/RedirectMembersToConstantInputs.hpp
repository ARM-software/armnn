//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "Optimization.hpp"

#include <armnn/utility/IgnoreUnused.hpp>
#include <armnn/utility/PolymorphicDowncast.hpp>

namespace armnn
{
namespace optimizations
{

class RedirectMembersToConstantInputsImpl
{
public:
    /// Search for layers with ConstantLayers as inputs. If the inputs are constant redirect the layers member
    /// variable for ConstTensors (e.g. m_weights) to the data stored in the ConstantLayer it is connected to.
    void Run(Graph& graph, Layer& layer) const
    {
        IgnoreUnused(graph);

        switch (layer.GetType())
        {
            case LayerType::BatchNormalization:
                break;
            case LayerType::Convolution2d:
                break;
            case LayerType::DepthwiseConvolution2d:
                break;
            case LayerType::DetectionPostProcess:
                break;
            case LayerType::FullyConnected:
                RedirectWeightsAndBiases<FullyConnectedLayer>(&layer);
                break;
            case LayerType::Lstm:
                break;
            case LayerType::TransposeConvolution2d:
                break;
            default:
                break;
        }
    }

protected:
    RedirectMembersToConstantInputsImpl()  = default;
    ~RedirectMembersToConstantInputsImpl() = default;

private:
    template <typename LayerT>
    static LayerT* RedirectWeightsAndBiases(Layer* layer)
    {
        LayerT* layerPtr = PolymorphicDowncast<LayerT*>(layer);

        // Loop through input slots to check for constant weights and biases layers.
        // Weights index = 1, Biases index = 2.
        for (unsigned int inputSlotIndex = 1; inputSlotIndex != layerPtr->GetNumInputSlots(); ++inputSlotIndex)
        {
            OutputSlot* outputSlot = layerPtr->GetInputSlot(inputSlotIndex).GetConnectedOutputSlot();
            if (outputSlot->GetOwningLayer().GetType() == LayerType::Constant)
            {
                // Get constant layer and redirect base layer member variables.
                ConstantLayer& constantLayer = dynamic_cast<ConstantLayer&>(outputSlot->GetOwningLayer());
                if (inputSlotIndex == 1)
                {
                    layerPtr->m_Weight = constantLayer.m_LayerOutput;
                }
                else if (inputSlotIndex == 2)
                {
                    layerPtr->m_Bias = constantLayer.m_LayerOutput;
                }
            }
        }

        return layerPtr;
    }
};

using RedirectMembersToConstantInputs = OptimizeForType<FullyConnectedLayer, RedirectMembersToConstantInputsImpl>;

} // namespace optimizations
} // namespace armnn
