//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//
#pragma once

#include "LayerWithParameters.hpp"

namespace armnn
{

class ScopedCpuTensorHandle;

class FullyConnectedLayer : public LayerWithParameters<FullyConnectedDescriptor>
{
public:
    std::unique_ptr<ScopedCpuTensorHandle> m_Weight;
    std::unique_ptr<ScopedCpuTensorHandle> m_Bias;

    virtual std::unique_ptr<IWorkload> CreateWorkload(const Graph&            graph,
                                                      const IWorkloadFactory& factory) const override;

    FullyConnectedLayer* Clone(Graph& graph) const override;

    void ValidateTensorShapesFromInputs() override;

protected:
    FullyConnectedLayer(const FullyConnectedDescriptor& param, const char* name);
    ~FullyConnectedLayer() = default;
};

} // namespace
