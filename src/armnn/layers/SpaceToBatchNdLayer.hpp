//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "LayerWithParameters.hpp"

namespace armnn
{

class SpaceToBatchNdLayer : public LayerWithParameters<SpaceToBatchNdDescriptor>
{
public:
    virtual std::unique_ptr<IWorkload> CreateWorkload(const Graph&            graph,
                                                      const IWorkloadFactory& factory) const override;

    SpaceToBatchNdLayer* Clone(Graph& graph) const override;

    std::vector<TensorShape> InferOutputShapes(const std::vector<TensorShape>& inputShapes) const override;

    void ValidateTensorShapesFromInputs() override;

protected:
    SpaceToBatchNdLayer(const SpaceToBatchNdDescriptor param, const char* name);
    ~SpaceToBatchNdLayer() = default;
};

} // namespace
