//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <Layer.hpp>

namespace armnn
{

class ScopedTensorHandle;

/// A layer that the constant data can be bound to.
class ConstantLayer : public Layer
{
public:
    /// Makes a workload for the Constant type.
    /// @param [in] graph The graph where this layer can be found.
    /// @param [in] factory The workload factory which will create the workload.
    /// @return A pointer to the created workload, or nullptr if not created.
    virtual std::unique_ptr<IWorkload> CreateWorkload(const IWorkloadFactory& factory) const override;

    /// Creates a dynamically-allocated copy of this layer.
    /// @param [in] graph The graph into which this layer is being cloned.
    ConstantLayer* Clone(Graph& graph) const override;

    /// Check if the input tensor shape(s)
    /// will lead to a valid configuration of @ref ConstantLayer
    /// @param [in] shapeInferenceMethod Indicates if output shape shall be overwritten or just validated.
    void ValidateTensorShapesFromInputs() override;

    /// By default returns inputShapes if the number of inputs are equal to number of outputs,
    /// otherwise infers the output shapes from given input shapes and layer properties.
    /// @param [in] inputShapes The input shapes layer has.
    /// @return a vector to the inferred output shape.
    std::vector<TensorShape> InferOutputShapes(const std::vector<TensorShape>& inputShapes) const override;

    /// Free up the constant source data stored by the layer.
    void ReleaseConstantData() override {}

    void ExecuteStrategy(IStrategy& strategy) const override;

    std::shared_ptr<ConstTensorHandle> m_LayerOutput;

protected:
    /// Constructor to create a ConstantLayer.
    /// @param [in] name Optional name for the layer.
    ConstantLayer(const char* name);

    /// Default destructor
    ~ConstantLayer() = default;

    /// Retrieve the handles to the constant values stored by the layer.
    // For API stability DO NOT ALTER order and add new members to the end of vector
    ImmutableConstantTensors GetConstantTensorsByRef() const override { return {m_LayerOutput}; }
};

} // namespace
