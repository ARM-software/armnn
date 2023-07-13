//
// Copyright © 2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "LayerWithParameters.hpp"

namespace armnn
{

class TileLayer : public LayerWithParameters<TileDescriptor>
{
public:
    /// Makes a workload for the Tile type.
    /// @param [in] graph The graph where this layer can be found.
    /// @param [in] factory The workload factory which will create the workload.
    /// @return A pointer to the created workload, or nullptr if not created.
    virtual std::unique_ptr<IWorkload> CreateWorkload(const IWorkloadFactory& factory) const override;

    /// Creates a dynamically-allocated copy of this layer.
    /// @param [in] graph The graph into which this layer is being cloned.
    TileLayer* Clone(Graph& graph) const override;

    /// Infers the output shapes from given input shapes and layer properties.
    /// @param [in] inputShapes The input shapes layer has.
    /// @return A vector to the inferred output shape.
    std::vector<TensorShape> InferOutputShapes(const std::vector<TensorShape>& inputShapes) const override;

    /// Check if the input tensor tile(s)
    /// will lead to a valid configuration of @ref TileLayer.
    /// @param [in] shapeInferenceMethod Indicates if output tile shall be overwritten or just validated.
    void ValidateTensorShapesFromInputs() override;

protected:
    /// Constructor to create a TileLayer.
    /// @param [in] name Optional name for the layer.
    TileLayer(const TileDescriptor& param, const char* name);

    /// Default destructor.
    ~TileLayer() = default;
};

} // namespace armnn
