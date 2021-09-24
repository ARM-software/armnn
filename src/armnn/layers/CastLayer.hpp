//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "Layer.hpp"

namespace armnn
{

/// This layer represents a cast operation
class CastLayer : public Layer
{
public:
    /// Makes a workload for the Cast type.
    /// @param [in] factory The workload factory which will create the workload.
    /// @return A pointer to the created workload, or nullptr if not created.
    virtual std::unique_ptr <IWorkload> CreateWorkload(const IWorkloadFactory &factory) const override;

    /// Creates a dynamically-allocated copy of this layer.
    /// @param [in] graph The graph into which this layer is being cloned.
    CastLayer* Clone(Graph& graph) const override;

    /// Check if the input tensor shape(s)
    /// will lead to a valid configuration of @ref ConvertFp16ToFp32Layer.
    /// @param [in] shapeInferenceMethod Indicates if output shape shall be overwritten or just validated.
    void ValidateTensorShapesFromInputs() override;

    ARMNN_NO_DEPRECATE_WARN_BEGIN
    void Accept(ILayerVisitor& visitor) const override;
    ARMNN_NO_DEPRECATE_WARN_END

protected:
    /// Constructor to create a CastLayer.
    CastLayer(const char *name);

    /// Default destructor
    ~CastLayer() = default;
};
} // namespace armnn