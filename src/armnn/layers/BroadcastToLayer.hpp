//
// Copyright © 2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "LayerWithParameters.hpp"

namespace armnn
{

    class BroadcastToLayer : public LayerWithParameters<BroadcastToDescriptor>
    {
    public:
        /// Makes a workload for the BroadcastTo type.
        /// @param [in] factory The workload factory which will create the workload.
        /// @return A pointer to the created workload, or nullptr if not created.
        virtual std::unique_ptr<IWorkload> CreateWorkload(const IWorkloadFactory& factory) const override;

        /// Creates a dynamically-allocated copy of this layer.
        /// @param [in] graph The graph into which this layer is being cloned.
        BroadcastToLayer* Clone(Graph& graph) const override;

        /// Infers the output shapes from given input shapes and layer properties.
        /// @param [in] inputShapes The input shapes layer has.
        /// @return A vector to the inferred output shape.
        std::vector<TensorShape> InferOutputShapes(const std::vector<TensorShape>& inputShapes) const override;

        /// Check if the input tensor BroadcastTo(s)
        /// will lead to a valid configuration of @ref BroadcastToLayer.
        void ValidateTensorShapesFromInputs() override;

        /// Execute Strategy on BroadcastTo layer
        /// @param [in] strategy The input strategy for the layer
        void ExecuteStrategy(IStrategy& strategy) const override;

    protected:
        /// Constructor to create a BroadcastToLayer.
        /// @param [in] param Parameters for the layer.
        /// @param [in] name Optional name for the layer.
        BroadcastToLayer(const BroadcastToDescriptor& param, const char* name);

        /// Default destructor.
        ~BroadcastToLayer() = default;
    };

} // namespace armnn