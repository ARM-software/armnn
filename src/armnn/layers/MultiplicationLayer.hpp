//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//
#pragma once

#include "ArithmeticBaseLayer.hpp"

namespace armnn
{

class MultiplicationLayer : public ArithmeticBaseLayer
{
public:
    virtual std::unique_ptr<IWorkload> CreateWorkload(const Graph& graph,
                                                      const IWorkloadFactory& factory) const override;

    MultiplicationLayer* Clone(Graph& graph) const override;

protected:
    MultiplicationLayer(const char* name);
    ~MultiplicationLayer() = default;
};

} // namespace
