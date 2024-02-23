//
// Copyright © 2024 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "LayerWithParameters.hpp"

namespace armnn
{

/// This layer represents a ScatterNd operator.
class ScatterNdLayer : public LayerWithParameters<ScatterNdDescriptor>
{
public:
    /// Makes a workload for the ScatterNd type.
    /// @param [in] factory The workload factory which will create the workload.
    /// @return A pointer to the created workload, or nullptr if not created.
    virtual std::unique_ptr<IWorkload> CreateWorkload(const IWorkloadFactory& factory) const override;

    /// Creates a dynamically-allocated copy of this layer.
    /// @param [in] graph The graph into which this layer is being cloned.
    ScatterNdLayer* Clone(Graph& graph) const override;

    /// Infers the output shapes from given input shapes and layer properties.
    /// @param [in] inputShapes The input shapes layer has.
    /// @return A vector to the inferred output shape.
    std::vector<TensorShape> InferOutputShapes(const std::vector<TensorShape>& inputShapes) const override;

    /// Check if the input tensor shape(s)
    /// will lead to a valid configuration of @ref ScatterNdLayer.
    void ValidateTensorShapesFromInputs() override;

protected:
    /// Constructor to create a ScatterNdLayer.
    /// @param [in] name Optional name for the layer.
    ScatterNdLayer(const ScatterNdDescriptor& param, const char* name);

    /// Default destructor
    ~ScatterNdLayer() = default;
};

} // namespace armnn
