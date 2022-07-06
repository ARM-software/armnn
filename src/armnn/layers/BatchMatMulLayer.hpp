//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "LayerWithParameters.hpp"

namespace armnn
{

class BatchMatMulLayer : public LayerWithParameters<BatchMatMulDescriptor>
{
public:
    /// Makes a workload for the BatchMatMul type.
    /// @param [in] graph The graph where this layer can be found.
    /// @param [in] factory The workload factory which will create the workload.
    /// @return A pointer to the created workload, or nullptr if not created.
    virtual std::unique_ptr<IWorkload> CreateWorkload(const IWorkloadFactory &factory) const override;

    /// Creates a dynamically-allocated copy of this layer.
    /// @param [in] graph The graph into which this layer is being cloned.
    BatchMatMulLayer* Clone(Graph &graph) const override;

    /// Infers the output shape from the given input shapes.
    /// @param [in] inputShapes The vector of input shapes for BatchMatMul.
    /// @return A vector of inferred output shape.
    std::vector<TensorShape> InferOutputShapes(const std::vector<TensorShape>& inputShapes) const override;

    /// Check if the input tensor shapes
    /// will lead to a valid configuration of @ref BatchMatMulLayer.
    /// @param [in] shapeInferenceMethod Indicates if output shape shall be overwritten or just validated.
    void ValidateTensorShapesFromInputs() override;

protected:
    /// Constructor to create a BatchMatMulLayer.
    /// @param [in] param BatchMatMulDescriptor to configure optional parameters for batch matrix multiplication
    /// @param [in] name Optional name for the layer
    BatchMatMulLayer(const BatchMatMulDescriptor& param, const char* name);

    /// Default destructor
    ~BatchMatMulLayer() = default;
};

}