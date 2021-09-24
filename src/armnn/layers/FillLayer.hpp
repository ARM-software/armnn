//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include "LayerWithParameters.hpp"

namespace armnn
{

/// This layer represents a fill operation.
class FillLayer : public LayerWithParameters<FillDescriptor>
{
public:
    /// Makes a workload for the Fill layer.
    /// @param [in] factory The workload factory which will create the workload.
    /// @return A pointer to the created workload, or nullptr if not created.
    virtual std::unique_ptr<IWorkload> CreateWorkload(const IWorkloadFactory& factory) const override;

    /// Creates a dynamically-allocated copy of this layer.
    /// @param [in] graph The graph into which this layer is being cloned.
    FillLayer* Clone(Graph& graph) const override;

    /// Check if the input tensor shape(s)
    /// will lead to a valid configuration of @ref FillLayer.
    /// @param [in] shapeInferenceMethod Indicates if output shape shall be overwritten or just validate.
    void ValidateTensorShapesFromInputs() override;

    ARMNN_NO_DEPRECATE_WARN_BEGIN
    void Accept(ILayerVisitor& visitor) const override;
    ARMNN_NO_DEPRECATE_WARN_END

protected:
    /// Constructor to create a FillLayer.
    /// @param [in] descriptor to configure the fill operation.
    /// @param [in] name Optional name for the layer.
    FillLayer(const FillDescriptor& descriptor, const char* name);

    /// Default destructor
    ~FillLayer() = default;
};

} // namespace
