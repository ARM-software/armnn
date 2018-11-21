//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include "LayerWithParameters.hpp"

namespace armnn
{

class StridedSliceLayer : public LayerWithParameters<StridedSliceDescriptor>
{
public:
    virtual std::unique_ptr<IWorkload> CreateWorkload(const Graph&            graph,
                                                      const IWorkloadFactory& factory) const override;

    StridedSliceLayer* Clone(Graph& graph) const override;

    std::vector<TensorShape> InferOutputShapes(const std::vector<TensorShape>& inputShapes) const override;

    void ValidateTensorShapesFromInputs() override;

protected:
    StridedSliceLayer(const StridedSliceDescriptor& param, const char* name);
    ~StridedSliceLayer() = default;
};

} // namespace
