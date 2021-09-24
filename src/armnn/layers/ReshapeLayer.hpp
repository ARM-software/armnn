//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include "LayerWithParameters.hpp"

#include <armnn/utility/PolymorphicDowncast.hpp>

namespace armnn
{

/// This layer represents a reshape operation.
class ReshapeLayer : public LayerWithParameters<ReshapeDescriptor>
{
public:
    /// Makes a workload for the Reshape type.
    /// @param [in] graph The graph where this layer can be found.
    /// @param [in] factory The workload factory which will create the workload.
    /// @return A pointer to the created workload, or nullptr if not created.
    virtual std::unique_ptr<IWorkload> CreateWorkload(const IWorkloadFactory& factory) const override;

    /// Creates a dynamically-allocated copy of this layer.
    /// @param [in] graph The graph into which this layer is being cloned.
    ReshapeLayer* Clone(Graph& graph) const override;

    /// Check if the input tensor shape(s)
    /// will lead to a valid configuration of @ref ReshapeLayer.
    /// @param [in] shapeInferenceMethod Indicates if output shape shall be overwritten or just validated.
    void ValidateTensorShapesFromInputs() override;

    /// By default returns inputShapes if the number of inputs are equal to number of outputs,
    /// otherwise infers the output shapes from given input shapes and layer properties.
    /// @param [in] inputShapes The input shapes layer has.
    /// @return A vector to the inferred output shape.
    std::vector<TensorShape> InferOutputShapes(const std::vector<TensorShape>& inputShapes) const override;

    /// Indicates if the other layer received is equal to this one.
    /// @param other The other layer to be compared with.
    /// @return true if other layer is equal to this false otherwise.
    bool IsEqual(const Layer& other) const
    {
        return (other.GetType() == LayerType::Reshape) &&
               m_Param.m_TargetShape == PolymorphicDowncast<const ReshapeLayer*>(&other)->m_Param.m_TargetShape;
    }

    ARMNN_NO_DEPRECATE_WARN_BEGIN
    void Accept(ILayerVisitor& visitor) const override;
    ARMNN_NO_DEPRECATE_WARN_END


protected:
    /// Constructor to create a ReshapeLayer.
    /// @param [in] param ReshapeDescriptor to configure the reshape operation.
    /// @param [in] name Optional name for the layer.
    ReshapeLayer(const ReshapeDescriptor& desc, const char* name);

    /// Default destructor
    ~ReshapeLayer() = default;
};

} // namespace
