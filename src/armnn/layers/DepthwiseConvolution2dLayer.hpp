//
// Copyright © 2017,2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include "LayerWithParameters.hpp"

namespace armnn
{

class ScopedTensorHandle;

/// This layer represents a depthwise convolution 2d operation.
class DepthwiseConvolution2dLayer : public LayerWithParameters<DepthwiseConvolution2dDescriptor>
{
public:

    /// Makes a workload for the DepthwiseConvolution2d type.
    /// @param [in] graph The graph where this layer can be found.
    /// @param [in] factory The workload factory which will create the workload.
    /// @return A pointer to the created workload, or nullptr if not created.
    virtual std::unique_ptr<IWorkload> CreateWorkload(const IWorkloadFactory& factory) const override;

    /// Creates a dynamically-allocated copy of this layer.
    /// @param [in] graph The graph into which this layer is being cloned.
    DepthwiseConvolution2dLayer* Clone(Graph& graph) const override;

    /// Check if the input tensor shape(s)
    /// will lead to a valid configuration of @ref DepthwiseConvolution2dLayer.
    /// @param [in] shapeInferenceMethod Indicates if output shape shall be overwritten or just validated.
    void ValidateTensorShapesFromInputs() override;

    /// By default returns inputShapes if the number of inputs are equal to number of outputs,
    /// otherwise infers the output shapes from given input shapes and layer properties.
    /// @param [in] inputShapes The input shapes layer has.
    /// @return A vector to the inferred output shape.
    std::vector<TensorShape> InferOutputShapes(const std::vector<TensorShape>& inputShapes) const override;

    void ExecuteStrategy(IStrategy& strategy) const override;

    void SerializeLayerParameters(ParameterStringifyFunction& fn) const override;

    /// This layer does not have any data stored, weights and bias are now stored in constant layers.
    /// We do not want to release the data in the constant layer, that is why we override with an empty function.
    void ReleaseConstantData() override {}

protected:
    /// Constructor to create a DepthwiseConvolution2dLayer.
    /// @param [in] param DepthwiseConvolution2dDescriptor to configure the depthwise convolution2d.
    /// @param [in] name Optional name for the layer.
    DepthwiseConvolution2dLayer(const DepthwiseConvolution2dDescriptor& param, const char* name);

    /// Default destructor
    ~DepthwiseConvolution2dLayer() = default;

    /// Retrieve the handles to the constant values connected to the layer.
    /// @return A vector of the constant tensors connected to the layer.
    ImmutableConstantTensors GetConstantTensorsByRef() const override;
};

} // namespace
