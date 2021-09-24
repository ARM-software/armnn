//
// Copyright © 2019 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "LayerWithParameters.hpp"

namespace armnn
{

/// This layer represents an unknown operation in the input graph.
class StandInLayer : public LayerWithParameters<StandInDescriptor>
{
public:
    /// Empty implementation explictly does NOT create a workload. Throws Exception if called.
    /// @param [in] graph The graph where this layer can be found.
    /// @param [in] factory The workload factory which will create the workload.
    /// @return Does not return anything. Throws Exception if called.
    virtual std::unique_ptr<IWorkload>CreateWorkload(const IWorkloadFactory& factory) const override;

    /// Creates a dynamically-allocated copy of this layer.
    /// @param [in] graph The graph into which this layer is being cloned.
    StandInLayer* Clone(Graph& graph) const override;

    /// Check if the input tensor shape(s)
    /// Does nothing since cannot validate any properties of this layer.
    /// @param [in] shapeInferenceMethod Indicates if output shape shall be overwritten or just validated.
    void ValidateTensorShapesFromInputs() override;

    /// Empty implementation that throws Exception if called.
    /// otherwise infers the output shapes from given input shapes and layer properties.
    /// @param [in] inputShapes The input shapes layer has.
    /// @return Does not return anything. Throws Exception if called.
    std::vector<TensorShape> InferOutputShapes(const std::vector<TensorShape>& inputShapes) const override;

    /// Accepts a visitor object and calls VisitStandInLayer() method.
    /// @param visitor The visitor on which to call VisitStandInLayer() method.
    ARMNN_NO_DEPRECATE_WARN_BEGIN
    void Accept(ILayerVisitor& visitor) const override;
    ARMNN_NO_DEPRECATE_WARN_END


protected:
    /// Constructor to create a StandInLayer.
    /// @param [in] param StandInDescriptor to configure the stand-in operation.
    /// @param [in] name Optional name for the layer.
    StandInLayer(const StandInDescriptor& param, const char* name);

    /// Default destructor
    ~StandInLayer() = default;
};

} //namespace armnn




