//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//
#pragma once

#include <Layer.hpp>

namespace armnn
{

class OutputLayer : public BindableLayer
{
public:
    virtual std::unique_ptr<IWorkload> CreateWorkload(const Graph&            graph,
                                                      const IWorkloadFactory& factory) const override;
    virtual void CreateTensorHandles(Graph& graph, const IWorkloadFactory& factory) override
    {
        boost::ignore_unused(graph, factory);
    }

    OutputLayer* Clone(Graph& graph) const override;

    void ValidateTensorShapesFromInputs() override;

protected:
    OutputLayer(LayerBindingId id, const char* name);
    ~OutputLayer() = default;
};

} // namespace
