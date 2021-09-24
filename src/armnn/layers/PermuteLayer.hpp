//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include "LayerWithParameters.hpp"

#include <armnn/utility/PolymorphicDowncast.hpp>

namespace armnn
{

/// This layer represents a permutation operation.
class PermuteLayer : public LayerWithParameters<PermuteDescriptor>
{
public:
    /// Makes a workload for the Permute type.
    /// @param [in] graph The graph where this layer can be found.
    /// @param [in] factory The workload factory which will create the workload.
    /// @return A pointer to the created workload, or nullptr if not created.
    virtual std::unique_ptr<IWorkload> CreateWorkload(const IWorkloadFactory& factory) const override;

    /// Creates a dynamically-allocated copy of this layer.
    /// @param [in] graph The graph into which this layer is being cloned.
    PermuteLayer* Clone(Graph& graph) const override;

    /// Check if the input tensor shape(s)
    /// will lead to a valid configuration of @ref PermuteLayer.
    /// @param [in] shapeInferenceMethod Indicates if output shape shall be overwritten or just validated.
    void ValidateTensorShapesFromInputs() override;

    /// By default returns inputShapes if the number of inputs are equal to number of outputs,
    /// otherwise infers the output shapes from given input shapes and layer properties.
    /// @param [in] inputShapes The input shapes layer has.
    /// @return A vector to the inferred output shape.
    std::vector<TensorShape> InferOutputShapes(const std::vector<TensorShape>& inputShapes) const override;

    /// @return a permutation vector represents the memory layout of the tensor elements.
    const PermutationVector& GetPermutation() const
    {
        return m_Param.m_DimMappings;
    }

    /// Indicates if the other layer received is inverse of this one.
    /// @param other The other layer to be compared with.
    /// @return true if other layer is inverse of this false otherwise.
    bool IsInverse(const Layer& other) const
    {
        return (other.GetType() == LayerType::Permute) &&
            GetPermutation().IsInverse(PolymorphicDowncast<const PermuteLayer*>(&other)->GetPermutation());
    }

    /// Indicates if the other layer received is equal to this one.
    /// @param other The other layer to be compare with.
    /// @return true if other layer is equal to this false otherwise.
    bool IsEqual(const Layer& other) const
    {
        return (other.GetType() == LayerType::Permute) &&
               GetPermutation().IsEqual(PolymorphicDowncast<const PermuteLayer*>(&other)->GetPermutation());
    }

    ARMNN_NO_DEPRECATE_WARN_BEGIN
    void Accept(ILayerVisitor& visitor) const override;
    ARMNN_NO_DEPRECATE_WARN_END


protected:
    /// Constructor to create a PermuteLayer.
    /// @param [in] param PermuteDescriptor to configure the permute operation.
    /// @param [in] name Optional name for the layer.
    PermuteLayer(const PermuteDescriptor& param, const char* name);

    /// Default destructor
    ~PermuteLayer() = default;
};

} // namespace
