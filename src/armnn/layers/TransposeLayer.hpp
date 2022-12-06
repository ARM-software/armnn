//
// Copyright © 2020,2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include "LayerWithParameters.hpp"

#include <armnn/utility/PolymorphicDowncast.hpp>

namespace armnn
{

/// This layer represents a transpose operation.
class TransposeLayer : public LayerWithParameters<TransposeDescriptor>
{
public:
    /// Makes a workload for the Transpose type.
    /// @param [in] factory The workload factory which will create the workload.
    /// @return A pointer to the created workload, or nullptr if not created.
    virtual std::unique_ptr<IWorkload> CreateWorkload(const IWorkloadFactory& factory) const override;

    /// Creates a dynamically-allocated copy of this layer.
    /// @param [in] graph The graph into which this layer is being cloned.
    TransposeLayer* Clone(Graph& graph) const override;

    /// Check if the input tensor shape(s)
    /// will lead to a valid configuration of @ref TransposeLayer.
    void ValidateTensorShapesFromInputs() override;

    /// Infers the output shapes from given input shapes and the permutation vector.
    /// @param [in] inputShapes The input shapes layer has.
    /// @return A vector to the inferred output shape.
    std::vector<TensorShape> InferOutputShapes(const std::vector<TensorShape>& inputShapes) const override;

    /// @return a permutation vector describing the permutation for the dimensions of the input tensor.
    const PermutationVector& GetPermutation() const
    {
        return m_Param.m_DimMappings;
    }

    /// Indicates if the other layer received is inverse of this one.
    /// @param [in] other The other layer to be compared with.
    /// @return true if other layer is inverse of this false otherwise.
    bool IsInverse(const Layer& other) const
    {
        return (other.GetType() == LayerType::Transpose) &&
            GetPermutation().IsInverse(PolymorphicDowncast<const TransposeLayer*>(&other)->GetPermutation());
    }

    /// Indicates if the other layer received is equal to this one.
    /// @param [in] other The other layer to be compare with.
    /// @return true if other layer is equal to this false otherwise.
    bool IsEqual(const Layer& other) const
    {
        return (other.GetType() == LayerType::Transpose) &&
               GetPermutation().IsEqual(PolymorphicDowncast<const TransposeLayer*>(&other)->GetPermutation());
    }

    void ExecuteStrategy(IStrategy& strategy) const override;


protected:
    /// Constructor to create a TransposeLayer.
    /// @param [in] param TransposeDescriptor to configure the permute operation.
    /// @param [in] name Optional name for the layer.
    TransposeLayer(const TransposeDescriptor& param, const char* name);

    /// Default destructor
    ~TransposeLayer() = default;
};

} // namespace
