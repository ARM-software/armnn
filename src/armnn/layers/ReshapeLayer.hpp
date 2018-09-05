//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include "LayerWithParameters.hpp"

namespace armnn
{

class ReshapeLayer : public LayerWithParameters<ReshapeDescriptor>
{
public:
    virtual std::unique_ptr<IWorkload> CreateWorkload(const Graph& graph,
        const IWorkloadFactory& factory) const override;

    ReshapeLayer* Clone(Graph& graph) const override;

    void ValidateTensorShapesFromInputs() override;
    std::vector<TensorShape> InferOutputShapes(const std::vector<TensorShape>& inputShapes) const override;

    bool IsEqual(const Layer& other) const
    {
        return (other.GetType() == LayerType::Reshape) &&
               m_Param.m_TargetShape == boost::polymorphic_downcast<const ReshapeLayer*>(&other)->m_Param.m_TargetShape;
    }

protected:
    ReshapeLayer(const ReshapeDescriptor& desc, const char* name);
    ~ReshapeLayer() = default;
};

} // namespace
