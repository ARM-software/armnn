//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//
#pragma once

#include "LayerWithParameters.hpp"

namespace armnn
{

class ScopedCpuTensorHandle;

class Convolution2dLayer : public LayerWithParameters<Convolution2dDescriptor>
{
public:
    std::unique_ptr<ScopedCpuTensorHandle> m_Weight;
    std::unique_ptr<ScopedCpuTensorHandle> m_Bias;

    virtual std::unique_ptr<IWorkload> CreateWorkload(const Graph&            graph,
                                                      const IWorkloadFactory& factory) const override;

    Convolution2dLayer* Clone(Graph& graph) const override;

    void ValidateTensorShapesFromInputs() override;

protected:
    Convolution2dLayer(const Convolution2dDescriptor& param, const char* name);
    ~Convolution2dLayer() = default;
};

} // namespace
