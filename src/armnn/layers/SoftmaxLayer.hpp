//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include "LayerWithParameters.hpp"

namespace armnn
{

/// This layer represents a softmax operation.
class SoftmaxLayer : public LayerWithParameters<SoftmaxDescriptor>
{
public:
    /// Makes a workload for the Softmax type.
    /// @param [in] graph The graph where this layer can be found.
    /// @param [in] factory The workload factory which will create the workload.
    /// @return A pointer to the created workload, or nullptr if not created.
    virtual std::unique_ptr<IWorkload> CreateWorkload(const IWorkloadFactory& factory) const override;

    /// Creates a dynamically-allocated copy of this layer.
    /// @param [in] graph The graph into which this layer is being cloned.
    SoftmaxLayer* Clone(Graph& graph) const override;

    /// Check if the input tensor shape(s)
    /// will lead to a valid configuration of @ref SoftmaxLayer.
    /// @param [in] shapeInferenceMethod Indicates if output shape shall be overwritten or just validated.
    void ValidateTensorShapesFromInputs() override;

    ARMNN_NO_DEPRECATE_WARN_BEGIN
    void Accept(ILayerVisitor& visitor) const override;
    ARMNN_NO_DEPRECATE_WARN_END


protected:
    /// Constructor to create a SoftmaxLayer.
    /// @param [in] param SoftmaxDescriptor to configure the softmax operation.
    /// @param [in] name Optional name for the layer.
    SoftmaxLayer(const SoftmaxDescriptor& param, const char* name);

    /// Default destructor
    ~SoftmaxLayer() = default;
};

} // namespace
