//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
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

using PreCompiledObjectDeleter = std::function<void(const void*)>;
using PreCompiledObjectPtr = std::unique_ptr<void, PreCompiledObjectDeleter>;

class PreCompiledLayer : public LayerWithParameters<PreCompiledDescriptor>
{
public:
    PreCompiledLayer(const PreCompiledDescriptor& param, const char* name);
    ~PreCompiledLayer();

    virtual std::unique_ptr<IWorkload> CreateWorkload(const IWorkloadFactory& factory) const override;

    PreCompiledLayer* Clone(Graph &graph) const override;

    void ValidateTensorShapesFromInputs() override;

    void SetPreCompiledObject(PreCompiledObjectPtr preCompiledObject);

    void ExecuteStrategy(IStrategy& strategy) const override;

private:
    PreCompiledLayer(const PreCompiledLayer& other) = delete;
    PreCompiledLayer& operator=(const PreCompiledLayer& other) = delete;

    std::shared_ptr<void> m_PreCompiledObject;
};

} // namespace armnn
