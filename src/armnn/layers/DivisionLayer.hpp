//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//
#pragma once

#include <Layer.hpp>

namespace armnn
{

class DivisionLayer : public Layer
{
public:
    virtual std::unique_ptr<IWorkload> CreateWorkload(const Graph& graph,
                                                      const IWorkloadFactory& factory) const override;

    DivisionLayer* Clone(Graph& graph) const override;

    void ValidateTensorShapesFromInputs() override;
    std::vector<TensorShape> InferOutputShapes(const std::vector<TensorShape>& inputShapes) const override;

protected:
    DivisionLayer(const char* name);
    ~DivisionLayer() = default;
};

} // namespace
