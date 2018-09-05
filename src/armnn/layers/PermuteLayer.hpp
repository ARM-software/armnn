//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include "LayerWithParameters.hpp"

namespace armnn
{

class PermuteLayer : public LayerWithParameters<PermuteDescriptor>
{
public:
    virtual std::unique_ptr<IWorkload> CreateWorkload(const Graph&            graph,
                                                      const IWorkloadFactory& factory) const override;

    PermuteLayer* Clone(Graph& graph) const override;

    void ValidateTensorShapesFromInputs() override;
    std::vector<TensorShape> InferOutputShapes(const std::vector<TensorShape>& inputShapes) const override;

    const PermutationVector& GetPermutation() const
    {
        return m_Param.m_DimMappings;
    }

    bool IsInverse(const Layer& other) const
    {
        return (other.GetType() == LayerType::Permute) &&
            GetPermutation().IsInverse(boost::polymorphic_downcast<const PermuteLayer*>(&other)->GetPermutation());
    }

    bool IsEqual(const Layer& other) const
    {
        return (other.GetType() == LayerType::Permute) &&
               GetPermutation().IsEqual(boost::polymorphic_downcast<const PermuteLayer*>(&other)->GetPermutation());
    }

protected:
    PermuteLayer(const PermuteDescriptor& param, const char* name);
    ~PermuteLayer() = default;
};

} // namespace
