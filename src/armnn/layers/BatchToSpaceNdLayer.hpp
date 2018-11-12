//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include "LayerWithParameters.hpp"

namespace armnn
{

class BatchToSpaceNdLayer : public LayerWithParameters<BatchToSpaceNdDescriptor>
{
public:
    virtual std::unique_ptr<IWorkload> CreateWorkload(const Graph&            graph,
                                                      const IWorkloadFactory& factory) const override;

    BatchToSpaceNdLayer* Clone(Graph& graph) const override;

    void ValidateTensorShapesFromInputs() override;

    std::vector<TensorShape> InferOutputShapes(const std::vector<TensorShape>& inputShapes) const override;

protected:
    BatchToSpaceNdLayer(const BatchToSpaceNdDescriptor& param, const char* name);
    ~BatchToSpaceNdLayer() = default;
};

} // namespace
