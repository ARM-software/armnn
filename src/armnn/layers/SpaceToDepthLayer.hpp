//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "LayerWithParameters.hpp"

namespace armnn
{

/// This layer represents a SpaceToDepth operation.
class SpaceToDepthLayer : public LayerWithParameters<SpaceToDepthDescriptor>
{
public:
    /// Makes a workload for the SpaceToDepth type.
    /// @param [in] graph The graph where this layer can be found.
    /// @param [in] factory The workload factory which will create the workload.
    /// @return A pointer to the created workload, or nullptr if not created.
    virtual std::unique_ptr<IWorkload> CreateWorkload(const IWorkloadFactory& factory) const override;

    /// Creates a dynamically-allocated copy of this layer.
    /// @param [in] graph The graph into which this layer is being cloned.
    SpaceToDepthLayer* Clone(Graph& graph) const override;

    /// By default returns inputShapes if the number of inputs are equal to number of outputs,
    /// otherwise infers the output shapes from given input shapes and layer properties.
    /// @param [in] inputShapes The input shapes layer has.
    /// @return A vector to the inferred output shape.
    std::vector<TensorShape> InferOutputShapes(const std::vector<TensorShape>& inputShapes) const override;

    /// Check if the input tensor shape(s)
    /// will lead to a valid configuration of @ref SpaceToDepthLayer.
    /// @param [in] shapeInferenceMethod Indicates if output shape shall be overwritten or just validated.
    void ValidateTensorShapesFromInputs() override;

    void ExecuteStrategy(IStrategy& strategy) const override;


protected:
    /// Constructor to create a SpaceToDepthLayer.
    /// @param [in] param SpaceToDepthDescriptor to configure the SpaceToDepthLayer operation.
    /// @param [in] name Optional name for the layer.
    SpaceToDepthLayer(const SpaceToDepthDescriptor param, const char* name);

    /// Default destructor
    ~SpaceToDepthLayer() = default;
};

} // namespace armnn
