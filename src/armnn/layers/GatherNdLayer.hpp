//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "Layer.hpp"

namespace armnn
{

/// This layer represents a GatherNd operator.
class GatherNdLayer : public Layer
{
public:
    /// Makes a workload for the Gather type.
    /// @param [in] factory The workload factory which will create the workload.
    /// @return A pointer to the created workload, or nullptr if not created.
    virtual std::unique_ptr<IWorkload> CreateWorkload(const IWorkloadFactory& factory) const override;

    /// Creates a dynamically-allocated copy of this layer.
    /// @param [in] graph The graph into which this layer is being cloned.
    GatherNdLayer* Clone(Graph& graph) const override;

    /// Infers the output shapes from given input shapes and layer properties.
    /// @param [in] inputShapes The input shapes layer has.
    /// @return A vector to the inferred output shape.
    std::vector<TensorShape> InferOutputShapes(const std::vector<TensorShape>& inputShapes) const override;

    /// Check if the input tensor shape(s)
    /// will lead to a valid configuration of @ref GatherNdLayer.
    void ValidateTensorShapesFromInputs() override;

protected:
    /// Constructor to create a GatherNdLayer.
    /// @param [in] name Optional name for the layer.
    GatherNdLayer(const char* name);

    /// Default destructor
    ~GatherNdLayer() = default;
};

} // namespace armnn
