//
// Copyright © 2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "LayerWithParameters.hpp"

namespace armnn
{

/// This layer represents a elementwiseBinary operation.
class ElementwiseBinaryLayer : public LayerWithParameters<ElementwiseBinaryDescriptor>
{
public:
    /// Makes a workload for the elementwiseBinary type
    /// @param [in] graph The graph where this layer can be found
    /// @param [in] factory The workload factory which will create the workload
    /// @return A pointer to the created workload, or nullptr if not created
    virtual std::unique_ptr<IWorkload> CreateWorkload(const IWorkloadFactory& factory) const override;

    /// Creates a dynamically-allocated copy of this layer
    /// @param [in] graph The graph into which this layer is being cloned
    ElementwiseBinaryLayer* Clone(Graph& graph) const override;

    /// Returns inputShapes by default.
    /// @param [in] inputShapes The input shapes layer has.
    /// @return A vector to the inferred output shape.
    std::vector<TensorShape> InferOutputShapes(const std::vector<TensorShape>& inputShapes) const override;

    /// Check if the input tensor shape(s) will lead to a valid configuration
    /// of @ref ElementwiseBinaryLayer
    void ValidateTensorShapesFromInputs() override;

    void ExecuteStrategy(IStrategy& strategy) const override;

protected:
    /// Constructor to create a ElementwiseBinaryLayer
    /// @param [in] param ElementwiseBinaryDescriptor to configure the ElementwiseBinaryLayer
    /// @param [in] name Optional name for the layer
    ElementwiseBinaryLayer(const ElementwiseBinaryDescriptor& param, const char* name);

    /// Default destructor
    ~ElementwiseBinaryLayer() = default;
};

} // namespace armnn
