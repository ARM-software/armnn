//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include "LayerWithParameters.hpp"

namespace armnn
{

class MergerLayer : public LayerWithParameters<OriginsDescriptor>
{
public:
    virtual std::unique_ptr<IWorkload> CreateWorkload(const Graph&            graph,
                                                      const IWorkloadFactory& factory) const override;
    virtual void CreateTensorHandles(Graph& graph, const IWorkloadFactory& factory) override;

    MergerLayer* Clone(Graph& graph) const override;

    void ValidateTensorShapesFromInputs() override;
    std::vector<TensorShape> InferOutputShapes(const std::vector<TensorShape>& inputShapes) const override;

protected:
    MergerLayer(const OriginsDescriptor& param, const char* name);
    ~MergerLayer() = default;
};

} // namespace
