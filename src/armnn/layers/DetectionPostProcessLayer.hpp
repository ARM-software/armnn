//
// Copyright © 2017-2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "LayerWithParameters.hpp"

namespace armnn
{

class ScopedTensorHandle;

/// This layer represents a detection postprocess operator.
class DetectionPostProcessLayer : public LayerWithParameters<DetectionPostProcessDescriptor>
{
public:
    /// A unique pointer to store Anchor values.
    std::shared_ptr<ConstTensorHandle> m_Anchors;

    /// Makes a workload for the DetectionPostProcess type.
    /// @param [in] graph The graph where this layer can be found.
    /// @param [in] factory The workload factory which will create the workload.
    /// @return A pointer to the created workload, or nullptr if not created.
    virtual std::unique_ptr<IWorkload> CreateWorkload(const IWorkloadFactory& factory) const override;

    /// Creates a dynamically-allocated copy of this layer.
    /// @param [in] graph The graph into which this layer is being cloned.
    DetectionPostProcessLayer* Clone(Graph& graph) const override;

    /// Check if the input tensor shape(s)
    /// will lead to a valid configuration of @ref DetectionPostProcessLayer.
    /// @param [in] shapeInferenceMethod Indicates if output shape shall be overwritten or just validated.
    void ValidateTensorShapesFromInputs() override;

    /// The model does not specify the output shapes. The output shapes are calculated from the max_detection and
    /// max_classes_per_detection parameters in the DetectionPostProcessDescriptor.
    /// @param [in] inputShapes The input shapes layer has. These are ignored for DetectionPostProcessLayer.
    /// @return A vector to the inferred output shape.
    std::vector<TensorShape> InferOutputShapes(const std::vector<TensorShape>& inputShapes) const override;

    void ExecuteStrategy(IStrategy& strategy) const override;

protected:
    /// Constructor to create a DetectionPostProcessLayer.
    /// @param [in] param DetectionPostProcessDescriptor to configure the detection postprocess.
    /// @param [in] name Optional name for the layer.
    DetectionPostProcessLayer(const DetectionPostProcessDescriptor& param, const char* name);

    /// Default destructor
    ~DetectionPostProcessLayer() = default;

    /// Retrieve the handles to the constant values stored by the layer.
    /// @return A vector of the constant tensors stored by this layer.
    ImmutableConstantTensors GetConstantTensorsByRef() const override;
};

} // namespace armnn

