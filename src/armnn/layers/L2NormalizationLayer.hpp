//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include "LayerWithParameters.hpp"

namespace armnn
{

class L2NormalizationLayer : public LayerWithParameters<L2NormalizationDescriptor>
{
public:
    virtual std::unique_ptr<IWorkload> CreateWorkload(const Graph& graph,
        const IWorkloadFactory& factory) const override;

    L2NormalizationLayer* Clone(Graph& graph) const override;

    void ValidateTensorShapesFromInputs() override;

protected:
    L2NormalizationLayer(const L2NormalizationDescriptor& param, const char* name);
    ~L2NormalizationLayer() = default;
};

} // namespace
