//
// Copyright © 2019 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "LayerWithParameters.hpp"

namespace armnn
{

/// This layer represents a comparison operation.
class ComparisonLayer : public LayerWithParameters<ComparisonDescriptor>
{
public:
    /// Makes a workload for the Comparison type
    /// @param [in] graph The graph where this layer can be found
    /// @param [in] factory The workload factory which will create the workload
    /// @return A pointer to the created workload, or nullptr if not created
    virtual std::unique_ptr<IWorkload> CreateWorkload(const IWorkloadFactory& factory) const override;

    /// Creates a dynamically-allocated copy of this layer
    /// @param [in] graph The graph into which this layer is being cloned
    ComparisonLayer* Clone(Graph& graph) const override;

    /// By default returns inputShapes if the number of inputs are equal to number of outputs,
    /// otherwise infers the output shapes from given input shapes and layer properties.
    /// @param [in] inputShapes The input shapes layer has.
    /// @return A vector to the inferred output shape.
    std::vector<TensorShape> InferOutputShapes(const std::vector<TensorShape>& inputShapes) const override;

    /// Check if the input tensor shape(s) will lead to a valid configuration
    /// of @ref ComparisonLayer
    /// @param [in] shapeInferenceMethod Indicates if output shape shall be overwritten or just validated.
    void ValidateTensorShapesFromInputs() override;

    ARMNN_NO_DEPRECATE_WARN_BEGIN
    void Accept(ILayerVisitor& visitor) const override;
    ARMNN_NO_DEPRECATE_WARN_END

protected:
    /// Constructor to create a ComparisonLayer
    /// @param [in] param ComparisonDescriptor to configure the ComparisonLayer
    /// @param [in] name Optional name for the layer
    ComparisonLayer(const ComparisonDescriptor& param, const char* name);

    /// Default destructor
    ~ComparisonLayer() = default;
};

} // namespace armnn
