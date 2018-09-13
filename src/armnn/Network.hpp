//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <armnn/DescriptorsFwd.hpp>
#include <armnn/LstmParams.hpp>
#include <armnn/TensorFwd.hpp>
#include <armnn/Types.hpp>

#include <armnn/INetwork.hpp>

#include <string>
#include <vector>
#include <memory>

#include "Layer.hpp"

namespace armnn
{
class Graph;

/// Private implementation of INetwork.
class Network final : public INetwork
{
public:
    Network();
    ~Network();

    const Graph& GetGraph() const { return *m_Graph; }

    Status PrintGraph() override;

    IConnectableLayer* AddInputLayer(LayerBindingId id, const char* name=nullptr) override;

    IConnectableLayer* AddConvolution2dLayer(const Convolution2dDescriptor& convolution2dDescriptor,
        const ConstTensor& weights,
        const char* name = nullptr) override;

    IConnectableLayer* AddConvolution2dLayer(const Convolution2dDescriptor& convolution2dDescriptor,
        const ConstTensor& weights,
        const ConstTensor& biases,
        const char* name = nullptr) override;

    IConnectableLayer* AddDepthwiseConvolution2dLayer(
        const DepthwiseConvolution2dDescriptor& convolution2dDescriptor,
        const ConstTensor&                      weights,
        const char*                             name = nullptr) override;

    IConnectableLayer* AddDepthwiseConvolution2dLayer(
        const DepthwiseConvolution2dDescriptor& convolution2dDescriptor,
        const ConstTensor&                      weights,
        const ConstTensor&                      biases,
        const char*                             name = nullptr) override;

    IConnectableLayer* AddFullyConnectedLayer(const FullyConnectedDescriptor& fullyConnectedDescriptor,
        const ConstTensor& weights,
        const char* name = nullptr) override;

    IConnectableLayer* AddFullyConnectedLayer(const FullyConnectedDescriptor& fullyConnectedDescriptor,
        const ConstTensor& weights,
        const ConstTensor& biases,
        const char* name = nullptr) override;

    IConnectableLayer* AddPermuteLayer(const PermuteDescriptor& permuteDescriptor,
                                       const char* name = nullptr) override;

    IConnectableLayer* AddPooling2dLayer(const Pooling2dDescriptor& pooling2dDescriptor,
        const char* name = nullptr) override;

    IConnectableLayer* AddActivationLayer(const ActivationDescriptor& activationDescriptor,
        const char* name = nullptr) override;

    IConnectableLayer* AddNormalizationLayer(const NormalizationDescriptor& normalizationDescriptor,
        const char* name = nullptr) override;

    IConnectableLayer* AddSoftmaxLayer(const SoftmaxDescriptor& softmaxDescriptor,
        const char* name = nullptr) override;

    IConnectableLayer* AddSplitterLayer(const ViewsDescriptor& splitterDescriptor,
        const char* name = nullptr) override;

    IConnectableLayer* AddMergerLayer(const OriginsDescriptor& mergerDescriptor,
        const char* name = nullptr) override;

    IConnectableLayer* AddAdditionLayer(const char* name = nullptr) override;

    IConnectableLayer* AddMultiplicationLayer(const char* name = nullptr) override;

    IConnectableLayer* AddBatchNormalizationLayer(const BatchNormalizationDescriptor& desc,
                                                  const ConstTensor&                  mean,
                                                  const ConstTensor&                  variance,
                                                  const ConstTensor&                  beta,
                                                  const ConstTensor&                  gamma,
                                                  const char*                         name = nullptr) override;

    IConnectableLayer* AddResizeBilinearLayer(const ResizeBilinearDescriptor& resizeDesc,
                                              const char* name = nullptr) override;

    IConnectableLayer* AddL2NormalizationLayer(const char* name = nullptr) override;

    IConnectableLayer* AddConstantLayer(const ConstTensor& input, const char* name = nullptr) override;

    IConnectableLayer* AddReshapeLayer(const ReshapeDescriptor& reshapeDescriptor,
                                       const char* name = nullptr) override;

    IConnectableLayer* AddFloorLayer(const char* name = nullptr) override;

    IConnectableLayer* AddOutputLayer(LayerBindingId id, const char* name = nullptr) override;

    IConnectableLayer* AddLstmLayer(const LstmDescriptor& descriptor,
                                    const LstmInputParams& params,
                                    const char* name = nullptr) override;

    IConnectableLayer* AddDivisionLayer(const char* name = nullptr) override;

    IConnectableLayer* AddSubtractionLayer(const char* name = nullptr) override;

    IConnectableLayer* AddMeanLayer(const MeanDescriptor& meanDescriptor, const char* name = nullptr) override;

private:
    IConnectableLayer* AddFullyConnectedLayerImpl(const FullyConnectedDescriptor& fullyConnectedDescriptor,
        const ConstTensor& weights,
        const ConstTensor* biases,
        const char* name);

    IConnectableLayer* AddConvolution2dLayerImpl(const Convolution2dDescriptor& convolution2dDescriptor,
        const ConstTensor& weights,
        const ConstTensor* biases,
        const char* name);

    IConnectableLayer* AddDepthwiseConvolution2dLayerImpl(
        const DepthwiseConvolution2dDescriptor& convolution2dDescriptor,
        const ConstTensor& weights,
        const ConstTensor* biases,
        const char* name);

    std::unique_ptr<Graph> m_Graph;
};

class OptimizedNetwork final : public IOptimizedNetwork
{
public:
    OptimizedNetwork(std::unique_ptr<Graph> graph);
    ~OptimizedNetwork();

    Status PrintGraph() override;
    Status SerializeToDot(std::ostream& stream) const override;

    Graph& GetGraph() { return *m_Graph; }

private:
    std::unique_ptr<Graph> m_Graph;
};

} // namespace armnn
