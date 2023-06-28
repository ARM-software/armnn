//
// Copyright © 2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "LayerWithParameters.hpp"

namespace armnn
{

    /// This layer represents a ReverseV2 operation.
    class ReverseV2Layer : public LayerWithParameters<ReverseV2Descriptor>
    {
    public:
        /// Makes a workload for the ReverseV2 type.
        /// @param [in] graph The graph where this layer can be found.
        /// @param [in] factory The workload factory which will create the workload.
        /// @return A pointer to the created workload, or nullptr if not created.
        virtual std::unique_ptr<IWorkload>CreateWorkload(const IWorkloadFactory& factory) const override;

        /// Creates a dynamically-allocated copy of this layer.
        /// @param [in] graph The graph into which this layer is being cloned.
        ReverseV2Layer* Clone(Graph& graph) const override;

        /// By default returns inputShapes if the number of inputs are equal to number of outputs,
        /// otherwise infers the output shapes from given input shapes and layer properties.
        /// @param [in] inputShapes The vector of input shapes for ReverseV2.
        /// @return A vector to the inferred output shape.

        /// Use the default Layer::InferOutputShape method
        // std::vector<TensorShape> InferOutputShapes(const std::vector<TensorShape>& inputShapes) const override;

        /// Check if the input tensor shape(s)
        /// will lead to a valid configuration of @ref ReverseV2Layer.
        void ValidateTensorShapesFromInputs() override;

    protected:
        /// Constructor to create a ReverseV2Layer.
        /// @param [in] param ReverseV2Descriptor to configure the ReverseV2 operation.
        /// @param [in] name Optional name for the layer.
        ReverseV2Layer(const ReverseV2Descriptor& param, const char* name);

        /// Default destructor
        ~ReverseV2Layer() = default;
    };

} // namespace armnn
