//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <Layer.hpp>

namespace armnn
{

/// NOTE: this is an abstract class to encapsulate the element wise operations, it does not implement:
/// std::unique_ptr<IWorkload> Layer::CreateWorkload(const IWorkloadFactory& factory) const = 0;
/// Layer* Clone(Graph& graph) const = 0;
class ElementwiseBaseLayer : public Layer
{
public:
    /// Check if the input tensor shape(s)
    /// will lead to a valid configuration of the element wise operation.
    /// @param [in] shapeInferenceMethod Indicates if output shape shall be overwritten or just validated.
    void ValidateTensorShapesFromInputs() override;

    /// By default returns inputShapes if the number of inputs are equal to number of outputs,
    /// otherwise infers the output shapes from given input shapes and layer properties.
    /// @param [in] inputShapes The input shapes layer has.
    /// @return A vector to the inferred output shape.
    std::vector<TensorShape> InferOutputShapes(const std::vector<TensorShape>& inputShapes) const override;

    void ExecuteStrategy(IStrategy& strategy) const override;

protected:
    /// @param numInputSlots The number of input slots for the layer.
    /// @param numOutputSlots The number of output slots for the layer.
    /// @param type The layer type.
    /// @param name Optional name for the layer.
    ElementwiseBaseLayer(unsigned int numInputSlots, unsigned int numOutputSlots, LayerType type, const char* name);

    /// Default destructor
    ~ElementwiseBaseLayer() = default;
};

} // namespace
