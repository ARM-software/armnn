//
// Copyright © 2019 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "LayerWithParameters.hpp"

namespace armnn
{

/// This layer represents a elementwiseUnary operation.
class ElementwiseUnaryLayer : public LayerWithParameters<ElementwiseUnaryDescriptor>
{
public:
    /// Makes a workload for the elementwiseUnary type
    /// @param [in] graph The graph where this layer can be found
    /// @param [in] factory The workload factory which will create the workload
    /// @return A pointer to the created workload, or nullptr if not created
    virtual std::unique_ptr<IWorkload> CreateWorkload(const IWorkloadFactory& factory) const override;

    /// Creates a dynamically-allocated copy of this layer
    /// @param [in] graph The graph into which this layer is being cloned
    ElementwiseUnaryLayer* Clone(Graph& graph) const override;

    /// Returns inputShapes by default.
    /// @param [in] inputShapes The input shapes layer has.
    /// @return A vector to the inferred output shape.
    std::vector<TensorShape> InferOutputShapes(const std::vector<TensorShape>& inputShapes) const override;

    /// Check if the input tensor shape(s) will lead to a valid configuration
    /// of @ref ElementwiseUnaryLayer
    /// @param [in] shapeInferenceMethod Indicates if output shape shall be overwritten or just validated.
    void ValidateTensorShapesFromInputs() override;

    ARMNN_NO_DEPRECATE_WARN_BEGIN
    void Accept(ILayerVisitor& visitor) const override;
    ARMNN_NO_DEPRECATE_WARN_END

protected:
    /// Constructor to create a ElementwiseUnaryLayer
    /// @param [in] param ElementwiseUnaryDescriptor to configure the ElementwiseUnaryLayer
    /// @param [in] name Optional name for the layer
    ElementwiseUnaryLayer(const ElementwiseUnaryDescriptor& param, const char* name);

    /// Default destructor
    ~ElementwiseUnaryLayer() = default;
};

} // namespace armnn
