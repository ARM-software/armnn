//
// Copyright © 2019 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include "LayerWithParameters.hpp"

namespace armnn
{

/// This layer represents an instance normalization operation.
class InstanceNormalizationLayer : public LayerWithParameters<InstanceNormalizationDescriptor>
{
public:
    /// Makes a workload for the InstanceNormalization type.
    /// @param [in] graph The graph where this layer can be found.
    /// @param [in] factory The workload factory which will create the workload.
    /// @return A pointer to the created workload, or nullptr if not created.
    virtual std::unique_ptr<IWorkload> CreateWorkload(const IWorkloadFactory& factory) const override;

    /// Creates a dynamically-allocated copy of this layer.
    /// @param [in] graph The graph into which this layer is being cloned.
    InstanceNormalizationLayer* Clone(Graph& graph) const override;

    /// Check if the input tensor shape(s)
    /// will lead to a valid configuration of @ref InstanceNormalizationLayer.
    /// @param [in] shapeInferenceMethod Indicates if output shape shall be overwritten or just validate.
    void ValidateTensorShapesFromInputs() override;

    ARMNN_NO_DEPRECATE_WARN_BEGIN
    void Accept(ILayerVisitor& visitor) const override;
    ARMNN_NO_DEPRECATE_WARN_END

protected:
    /// Constructor to create a InstanceNormalizationLayer.
    /// @param [in] param InstanceNormalizationDescriptor to configure the Instance normalization operation.
    /// @param [in] name Optional name for the layer.
    InstanceNormalizationLayer(const InstanceNormalizationDescriptor& param, const char* name);

    /// Default destructor
    ~InstanceNormalizationLayer() = default;
};

} // namespace
