//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//
#pragma once

#include <Layer.hpp>

namespace armnn
{

class ScopedCpuTensorHandle;

class ConstantLayer : public Layer
{
public:
    virtual std::unique_ptr<IWorkload> CreateWorkload(const Graph& graph,
        const IWorkloadFactory& factory) const override;

    ConstantLayer* Clone(Graph& graph) const override;

    void ValidateTensorShapesFromInputs() override;

protected:
    ConstantLayer(const std::shared_ptr<ScopedCpuTensorHandle>& input, const char* name);
    ~ConstantLayer() = default;

private:
    std::shared_ptr<ScopedCpuTensorHandle> m_LayerOutput;
};

} // namespace
