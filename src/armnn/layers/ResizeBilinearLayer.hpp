//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//
#pragma once

#include "LayerWithParameters.hpp"

namespace armnn
{

class ResizeBilinearLayer : public LayerWithParameters<ResizeBilinearDescriptor>
{
public:
    virtual std::unique_ptr<IWorkload>
        CreateWorkload(const Graph& graph, const IWorkloadFactory& factory) const override;

    ResizeBilinearLayer* Clone(Graph& graph) const override;

    void ValidateTensorShapesFromInputs() override;

protected:
    ResizeBilinearLayer(const ResizeBilinearDescriptor& param, const char* name);
    ~ResizeBilinearLayer() = default;
};

} // namespace
