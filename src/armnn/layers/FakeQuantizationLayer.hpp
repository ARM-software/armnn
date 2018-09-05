//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include "LayerWithParameters.hpp"

namespace armnn
{

class FakeQuantizationLayer : public LayerWithParameters<FakeQuantizationDescriptor>
{
public:
    virtual std::unique_ptr<IWorkload> CreateWorkload(const Graph&            graph,
                                                      const IWorkloadFactory& factory) const override;

    FakeQuantizationLayer* Clone(Graph& graph) const override;

    void ValidateTensorShapesFromInputs() override;

protected:
    FakeQuantizationLayer(const FakeQuantizationDescriptor& descriptor, const char* name);
    ~FakeQuantizationLayer() = default;
};

} // namespace
