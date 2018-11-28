//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include "ArithmeticBaseLayer.hpp"

namespace armnn
{

class MaximumLayer : public ArithmeticBaseLayer
{
public:
    virtual std::unique_ptr<IWorkload> CreateWorkload(const Graph& graph,
                                                      const IWorkloadFactory& factory) const override;

    MaximumLayer* Clone(Graph& graph) const override;

protected:
    MaximumLayer(const char* name);

    ~MaximumLayer() = default;
};

} // namespace
