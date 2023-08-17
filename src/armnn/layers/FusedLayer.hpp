//
// Copyright © 2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "LayerWithParameters.hpp"
#include <armnn/backends/WorkloadFactory.hpp>

#include <armnn/Descriptors.hpp>

#include <memory>
#include <functional>

namespace armnn
{

class FusedLayer : public LayerWithParameters<FusedDescriptor>
{
public:
    FusedLayer(const FusedDescriptor& param, const char* name);
    ~FusedLayer();

    virtual std::unique_ptr<IWorkload> CreateWorkload(const IWorkloadFactory& factory) const override;

    FusedLayer* Clone(Graph &graph) const override;

    void ValidateTensorShapesFromInputs() override;

    void ExecuteStrategy(IStrategy& strategy) const override;

private:
    FusedLayer(const FusedLayer& other) = delete;
    FusedLayer& operator=(const FusedLayer& other) = delete;
};

} // namespace armnn
