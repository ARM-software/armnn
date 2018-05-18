//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//
#pragma once

#include "LayerWithParameters.hpp"

namespace armnn
{

class ActivationLayer : public LayerWithParameters<ActivationDescriptor>
{
public:
    virtual std::unique_ptr<IWorkload> CreateWorkload(const Graph&            graph,
                                                      const IWorkloadFactory& factory) const override;

    ActivationLayer* Clone(Graph& graph) const override;

    void ValidateTensorShapesFromInputs() override;

protected:
    ActivationLayer(const ActivationDescriptor &param, const char* name);
    ~ActivationLayer() = default;
};

} // namespace
