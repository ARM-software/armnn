//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include "LayerWithParameters.hpp"

namespace armnn
{

class SoftmaxLayer : public LayerWithParameters<SoftmaxDescriptor>
{
public:
    virtual std::unique_ptr<IWorkload> CreateWorkload(const Graph&            graph,
                                                      const IWorkloadFactory& factory) const override;

    SoftmaxLayer* Clone(Graph& graph) const override;

    void ValidateTensorShapesFromInputs() override;

protected:
    SoftmaxLayer(const SoftmaxDescriptor& param, const char* name);
    ~SoftmaxLayer() = default;
};

} // namespace
