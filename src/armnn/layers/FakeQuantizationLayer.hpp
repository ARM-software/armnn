//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include "LayerWithParameters.hpp"

namespace armnn
{

/// This layer represents a fake quantization operation.
class FakeQuantizationLayer : public LayerWithParameters<FakeQuantizationDescriptor>
{
public:
    /// Makes a workload for the FakeQuantization type.
    /// @param [in] graph The graph where this layer can be found.
    /// @param [in] factory The workload factory which will create the workload.
    /// @return A pointer to the created workload, or nullptr if not created.
    virtual std::unique_ptr<IWorkload> CreateWorkload(const IWorkloadFactory& factory) const override;

    /// Creates a dynamically-allocated copy of this layer.
    /// @param [in] graph The graph into which this layer is being cloned.
    FakeQuantizationLayer* Clone(Graph& graph) const override;

    /// Check if the input tensor shape(s)
    /// will lead to a valid configuration of @ref FakeQuantizationLayer.
    /// @param [in] shapeInferenceMethod Indicates if output shape shall be overwritten or just validate.
    void ValidateTensorShapesFromInputs() override;


    void ExecuteStrategy(IStrategy& strategy) const override;

protected:
    /// Constructor to create a FakeQuantizationLayer.
    /// @param [in] param FakeQuantizationDescriptor to configure the fake quantization operation.
    /// @param [in] name Optional name for the layer.
    FakeQuantizationLayer(const FakeQuantizationDescriptor& descriptor, const char* name);

    /// Default destructor
    ~FakeQuantizationLayer() = default;
};

} // namespace
