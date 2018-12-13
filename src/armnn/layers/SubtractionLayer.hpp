//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "ElementwiseBaseLayer.hpp"

namespace armnn
{

class SubtractionLayer : public ElementwiseBaseLayer
{
public:
    virtual std::unique_ptr<IWorkload> CreateWorkload(const Graph& graph,
                                                      const IWorkloadFactory& factory) const override;

    SubtractionLayer* Clone(Graph& graph) const override;

protected:
    SubtractionLayer(const char* name);
    ~SubtractionLayer() = default;
};

} // namespace
