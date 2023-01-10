//
// Copyright © 2020-2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <Layer.hpp>

namespace armnn
{

class RankLayer : public Layer
{
public:
    /// Makes a workload for the Rank type.
    /// @param [in] factory The workload factory which will create the workload.
    /// @return A pointer to the created workload, or nullptr if not created.
    virtual std::unique_ptr<IWorkload> CreateWorkload(const IWorkloadFactory& factory) const override;

    /// Creates a dynamically-allocated copy of this layer.
    /// @param [in] graph The graph into which this layer is being cloned.
    Layer* Clone(Graph& graph) const override;

    /// Check if the input tensor shape(s)
    /// will lead to a valid configuration of @ref RankLayer.
    /// @param [in] shapeInferenceMethod Indicates if output shape shall be overwritten or just validated.
    void ValidateTensorShapesFromInputs() override;

    /// Rank returns a scalar specifying the rank of the input tensor. The rank of a tensor is the number
    /// of dimensions it has.
    /// @param [in] inputShapes The input shapes layer has. This is ignored for Rank.
    /// @return A vector to the inferred output shape.
    std::vector<TensorShape> InferOutputShapes(const std::vector<TensorShape>& inputShapes) const override;

    void ExecuteStrategy(IStrategy& strategy) const override;

protected:
    /// Constructor to create a RankLayer.
    /// @param [in] name Optional name for the layer.
    RankLayer(const char* name);

    /// Default destructor
    ~RankLayer() = default;
};

} //namespace armnn


