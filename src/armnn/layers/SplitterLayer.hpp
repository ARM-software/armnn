//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//
#pragma once

#include "LayerWithParameters.hpp"

namespace armnn
{

class SplitterLayer : public LayerWithParameters<ViewsDescriptor>
{
public:
    virtual std::unique_ptr<IWorkload> CreateWorkload(const Graph&            graph,
                                                      const IWorkloadFactory& factory) const override;
    virtual void CreateTensorHandles(Graph& graph, const IWorkloadFactory& factory) override;

    SplitterLayer* Clone(Graph& graph) const override;

    void ValidateTensorShapesFromInputs() override;

protected:
    SplitterLayer(const ViewsDescriptor& param, const char* name);
    ~SplitterLayer() = default;
};

} // namespace
