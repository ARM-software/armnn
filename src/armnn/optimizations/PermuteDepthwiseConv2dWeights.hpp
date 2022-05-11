//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include "Optimization.hpp"
#include "NetworkUtils.hpp"

#include <armnnUtils/Permute.hpp>

#include <fmt/format.h>

namespace armnn
{
namespace optimizations
{

class PermuteDepthwiseConv2dWeightsImpl
{
public:

    void Run(Graph& graph, Layer& layer) const
    {
        if (layer.GetType() == LayerType::DepthwiseConvolution2d)
        {
            AddPermuteLayer(graph, PolymorphicDowncast<DepthwiseConvolution2dLayer*>(&layer));
        }
    }

protected:
    PermuteDepthwiseConv2dWeightsImpl() = default;
    ~PermuteDepthwiseConv2dWeightsImpl() = default;

private:
    /// ArmNN format for weights for depthwise is [1, H, W, C] independently of the input/output layout
    ///
    /// ACL format for weights for depthwise is:
    /// - [1, H, W, C] for [N, H, W, C] input/output layout (matches with ArmNN)
    /// - [1, C, H, W] for [N, C, H, W] input/output layout
    ///
    /// Therefore ArmNN weights have to be permuted when input/output layout is [N, C, H, W] to pass them to ACL.
    static void AddPermuteLayer(Graph& graph, DepthwiseConvolution2dLayer* layer)
    {
        TensorInfo inputInfo = layer->GetInputSlot(0).GetConnectedOutputSlot()->GetTensorInfo();
        TensorInfo weightInfo = layer->GetInputSlot(1).GetConnectedOutputSlot()->GetTensorInfo();
        if (layer->GetParameters().m_DataLayout == armnn::DataLayout::NHWC)
        {
            // No permutation required. Input and weights data layouts are the same.
            return;
        }
        else if (layer->GetParameters().m_DataLayout == armnn::DataLayout::NCHW)
        {
            // Weights permutation required. Weights [N,H,W,C] and input [N,C,H,W] data layouts are different.
            // [ 1, H, W, I*M] --> [ 1, I * M, H, W ]
            PermutationVector permutationVector = { 0, 2, 3, 1 };
            TensorInfo weightsPermuted = armnnUtils::Permuted(weightInfo, permutationVector);

            // Inserts NewLayer so layers don't need to be re-sorted.
            PermuteLayer* permuteLayer =
                graph.InsertNewLayer<PermuteLayer>(layer->GetInputSlot(1),
                                                   PermuteDescriptor(permutationVector),
                                                   "permute_layer");
            permuteLayer->GetOutputSlot().SetTensorInfo(weightsPermuted);

            // Assign Permute BackendId to be the same as the Depthwise Conv2d BackendId.
            // Needed as backends have already been assigned at this stage.
            permuteLayer->SetBackendId(layer->GetBackendId());
        }
        else
        {
            throw InvalidArgumentException(fmt::format("Unknown data layout for tensor info conversion: {}",
                                                       GetDataLayoutName(layer->GetParameters().m_DataLayout)));
        }
    }
};

using PermuteDepthwiseConv2dWeights = OptimizeForType<Layer, PermuteDepthwiseConv2dWeightsImpl>;

} // namespace optimizations
} // namespace armnn
