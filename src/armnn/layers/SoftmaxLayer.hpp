//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//
#pragma once

#include "LayerWithParameters.hpp"

namespace armnn
{

class Pooling2dLayer : public LayerWithParameters<Pooling2dDescriptor>
{
public:
    virtual std::unique_ptr<IWorkload> CreateWorkload(const Graph&            graph,
                                                      const IWorkloadFactory& factory) const override;

    Pooling2dLayer* Clone(Graph& graph) const override;

    void ValidateTensorShapesFromInputs() override;

protected:
    Pooling2dLayer(const Pooling2dDescriptor& param, const char* name);
    ~Pooling2dLayer() = default;
};

} // namespace
