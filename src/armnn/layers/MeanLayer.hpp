//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "LayerWithParameters.hpp"

namespace armnn
{

class MeanLayer : public LayerWithParameters<MeanDescriptor>
{
public:
    virtual std::unique_ptr<IWorkload> CreateWorkload(const Graph&            graph,
                                                      const IWorkloadFactory& factory) const override;

    MeanLayer* Clone(Graph& graph) const override;

    void ValidateTensorShapesFromInputs() override;

protected:
    MeanLayer(const MeanDescriptor& param, const char* name);
    ~MeanLayer() = default;

};

}