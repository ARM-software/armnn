//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include "LayerWithParameters.hpp"

namespace armnn
{

/// This layer represents a normalization operation.
class NormalizationLayer : public LayerWithParameters<NormalizationDescriptor>
{
public:
    /// Makes a workload for the Normalization type.
    /// @param [in] graph The graph where this layer can be found.
    /// @param [in] factory The workload factory which will create the workload.
    /// @return A pointer to the created workload, or nullptr if not created.
    virtual std::unique_ptr<IWorkload> CreateWorkload(const IWorkloadFactory& factory) const override;

    /// Creates a dynamically-allocated copy of this layer.
    /// @param [in] graph The graph into which this layer is being cloned.
    NormalizationLayer* Clone(Graph& graph) const override;

    /// Check if the input tensor shape(s)
    /// will lead to a valid configuration of @ref NormalizationLayer.
    /// @param [in] shapeInferenceMethod Indicates if output shape shall be overwritten or just validated.
    void ValidateTensorShapesFromInputs() override;

    ARMNN_NO_DEPRECATE_WARN_BEGIN
    void Accept(ILayerVisitor& visitor) const override;
    ARMNN_NO_DEPRECATE_WARN_END


protected:
    /// Constructor to create a NormalizationLayer.
    /// @param [in] param NormalizationDescriptor to configure the normalization operation.
    /// @param [in] name Optional name for the layer.
    NormalizationLayer(const NormalizationDescriptor& param, const char* name);

    /// Default destructor
    ~NormalizationLayer() = default;
};

} // namespace
