//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include "LayerWithParameters.hpp"

namespace armnn
{

class Pooling2dLayer : public LayerWithParameters<Pooling2dDescriptor>
{
public:
    virtual std::unique_ptr<IWorkload> CreateWorkload(const Graph&            graph,
                                                      const IWorkloadFactory& factory) const override;

    Pooling2dLayer* Clone(Graph& graph) const override;

    void ValidateTensorShapesFromInputs() override;
    std::vector<TensorShape> InferOutputShapes(const std::vector<TensorShape>& inputShapes) const override;

protected:
    Pooling2dLayer(const Pooling2dDescriptor& param, const char* name);
    ~Pooling2dLayer() = default;
};

} // namespace
