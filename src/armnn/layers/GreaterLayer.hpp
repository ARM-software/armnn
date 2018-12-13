//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "ElementwiseBaseLayer.hpp"

namespace armnn
{

class GreaterLayer : public ElementwiseBaseLayer
{
public:
    virtual std::unique_ptr<IWorkload> CreateWorkload(const Graph& graph,
                                                      const IWorkloadFactory& factory) const override;

    GreaterLayer* Clone(Graph& graph) const override;

protected:
    GreaterLayer(const char* name);
    ~GreaterLayer() = default;
};

} //namespace armnn
