//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include "LayerWithParameters.hpp"

namespace armnn
{

/// This layer represents a merge operation.
class MergerLayer : public LayerWithParameters<OriginsDescriptor>
{
public:
    /// Makes a workload for the Merger type.
    /// @param [in] graph The graph where this layer can be found.
    /// @param [in] factory The workload factory which will create the workload.
    /// @return A pointer to the created workload, or nullptr if not created.
    virtual std::unique_ptr<IWorkload> CreateWorkload(const Graph&            graph,
                                                      const IWorkloadFactory& factory) const override;

    /// Set the outputs to be appropriate sub tensors of the input if sub tensors are supported
    /// otherwise creates tensor handlers.
    /// @param [in] graph The graph where this layer can be found.
    /// @param [in] factory The workload factory which will create the workload.
    virtual void CreateTensorHandles(Graph& graph, const IWorkloadFactory& factory) override;

    /// Creates a dynamically-allocated copy of this layer.
    /// @param [in] graph The graph into which this layer is being cloned.
    MergerLayer* Clone(Graph& graph) const override;

    /// Check if the input tensor shape(s)
    /// will lead to a valid configuration of @ref MergerLayer.
    void ValidateTensorShapesFromInputs() override;

    /// By default returns inputShapes if the number of inputs are equal to number of outputs,
    /// otherwise infers the output shapes from given input shapes and layer properties.
    /// @param [in] inputShapes The input shapes layer has.
    /// @return A vector to the inferred output shape.
    std::vector<TensorShape> InferOutputShapes(const std::vector<TensorShape>& inputShapes) const override;

protected:
    /// Constructor to create a MergerLayer.
    /// @param [in] param OriginsDescriptor to configure the merger operation.
    /// @param [in] name Optional name for the layer.
    MergerLayer(const OriginsDescriptor& param, const char* name);

    /// Default destructor
    ~MergerLayer() = default;
};

} // namespace
