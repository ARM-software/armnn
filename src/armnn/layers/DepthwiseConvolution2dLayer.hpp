//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//
#pragma once

#include "LayerWithParameters.hpp"

namespace armnn
{

class ScopedCpuTensorHandle;

class DepthwiseConvolution2dLayer : public LayerWithParameters<DepthwiseConvolution2dDescriptor>
{
public:
    std::unique_ptr<ScopedCpuTensorHandle> m_Weight;
    std::unique_ptr<ScopedCpuTensorHandle> m_Bias;

    virtual std::unique_ptr<IWorkload> CreateWorkload(const Graph& graph,
                                                      const IWorkloadFactory& factory) const override;

    DepthwiseConvolution2dLayer* Clone(Graph& graph) const override;

    void ValidateTensorShapesFromInputs() override;

    std::vector<TensorShape> InferOutputShapes(const std::vector<TensorShape>& inputShapes) const override;

protected:
    DepthwiseConvolution2dLayer(const DepthwiseConvolution2dDescriptor& param, const char* name);
    ~DepthwiseConvolution2dLayer() = default;

    ConstantTensors GetConstantTensorsByRef() override;
};

} // namespace
