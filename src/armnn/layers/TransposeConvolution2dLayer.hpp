//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include "LayerWithParameters.hpp"

namespace armnn
{

class ScopedTensorHandle;

/// This layer represents a 2D transpose convolution operation.
class TransposeConvolution2dLayer : public LayerWithParameters<TransposeConvolution2dDescriptor>
{
public:
    /// A unique pointer to store weight values.
    std::shared_ptr<ConstTensorHandle> m_Weight;
    /// A unique pointer to store bias values.
    std::shared_ptr<ConstTensorHandle> m_Bias;

    /// Makes a workload for the TransposeConvolution2d type.
    /// @param [in] graph The graph where this layer can be found.
    /// @param [in] factory The workload factory which will create the workload.
    /// @return A pointer to the created workload, or nullptr if not created.
    virtual std::unique_ptr<IWorkload> CreateWorkload(const IWorkloadFactory& factory) const override;

    /// Creates a dynamically-allocated copy of this layer.
    /// @param [in] graph The graph into which this layer is being cloned.
    TransposeConvolution2dLayer* Clone(Graph& graph) const override;

    /// Check if the input tensor shape(s)
    /// will lead to a valid configuration of @ref TransposeConvolution2dLayer.
    /// @param [in] shapeInferenceMethod Indicates if output shape shall be overwritten or just validated.
    void ValidateTensorShapesFromInputs() override;

    /// Infers the output shapes from given input shapes and layer properties.
    /// @param [in] inputShapes The input shapes the layer has.
    /// @return A vector of the inferred output shape.
    std::vector<TensorShape> InferOutputShapes(const std::vector<TensorShape>& inputShapes) const override;

    ARMNN_NO_DEPRECATE_WARN_BEGIN
    void Accept(ILayerVisitor& visitor) const override;
    ARMNN_NO_DEPRECATE_WARN_END


    void ExecuteStrategy(IStrategy& strategy) const override;

protected:
    /// Constructor to create a TransposeConvolution2dLayer.
    /// @param [in] param TransposeConvolution2dDescriptor to configure the 2D transpose convolution operation.
    /// @param [in] name Optional name for the layer.
    TransposeConvolution2dLayer(const TransposeConvolution2dDescriptor& param, const char* name);

    /// Default destructor
    ~TransposeConvolution2dLayer() = default;

    /// Retrieve the handles to the constant values stored by the layer.
    /// @return A vector of the constant tensors stored by this layer.
    ConstantTensors GetConstantTensorsByRef() override;
};

} // namespace armnn
