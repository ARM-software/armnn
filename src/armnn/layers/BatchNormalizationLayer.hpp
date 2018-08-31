//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//
#pragma once

#include "LayerWithParameters.hpp"

namespace armnn
{

class ScopedCpuTensorHandle;

class BatchNormalizationLayer : public LayerWithParameters<BatchNormalizationDescriptor>
{
public:
    std::unique_ptr<ScopedCpuTensorHandle> m_Mean;
    std::unique_ptr<ScopedCpuTensorHandle> m_Variance;
    std::unique_ptr<ScopedCpuTensorHandle> m_Beta;
    std::unique_ptr<ScopedCpuTensorHandle> m_Gamma;

    virtual std::unique_ptr<IWorkload> CreateWorkload(const Graph&            graph,
                                                      const IWorkloadFactory& factory) const override;

    BatchNormalizationLayer* Clone(Graph& graph) const override;

    void ValidateTensorShapesFromInputs() override;

protected:
    BatchNormalizationLayer(const BatchNormalizationDescriptor& param, const char* name);
    ~BatchNormalizationLayer() = default;

    ConstantTensors GetConstantTensorsByRef() override;
};

} // namespace
