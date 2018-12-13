//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "ElementwiseBaseLayer.hpp"

namespace armnn
{

class AdditionLayer : public ElementwiseBaseLayer
{
public:
    virtual std::unique_ptr<IWorkload> CreateWorkload(const Graph& graph,
                                                      const IWorkloadFactory& factory) const override;

    AdditionLayer* Clone(Graph& graph) const override;

protected:
    AdditionLayer(const char* name);
    ~AdditionLayer() = default;
};

} // namespace
