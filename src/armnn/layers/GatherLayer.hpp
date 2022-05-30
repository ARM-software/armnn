//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "LayerWithParameters.hpp"

namespace armnn
{

/// This layer represents a Gather operator.
class GatherLayer : public LayerWithParameters<GatherDescriptor>
{
public:
    /// Makes a workload for the Gather type.
    /// @param [in] graph The graph where this layer can be found.
    /// @param [in] factory The workload factory which will create the workload.
    /// @return A pointer to the created workload, or nullptr if not created.
    virtual std::unique_ptr<IWorkload> CreateWorkload(const IWorkloadFactory& factory) const override;

    /// Creates a dynamically-allocated copy of this layer.
    /// @param [in] graph The graph into which this layer is being cloned.
    GatherLayer* Clone(Graph& graph) const override;

    /// Infers the output shapes from given input shapes and layer properties.
    /// @param [in] inputShapes The input shapes layer has.
    /// @return A vector to the inferred output shape.
    std::vector<TensorShape> InferOutputShapes(const std::vector<TensorShape>& inputShapes) const override;

    /// Check if the input tensor shape(s).
    /// will lead to a valid configuration of @ref GatherLayer.
    /// @param [in] shapeInferenceMethod Indicates if output shape shall be overwritten or just validate.
    void ValidateTensorShapesFromInputs() override;

    void ExecuteStrategy(IStrategy& strategy) const override;

protected:
    /// Constructor to create a GatherLayer.
    /// @param [in] param GatherDescriptor to configure the stack operation.
    /// @param [in] name Optional name for the layer.
    GatherLayer(const GatherDescriptor& param, const char* name);

    /// Default destructor
    ~GatherLayer() = default;
};

} // namespace armnn
