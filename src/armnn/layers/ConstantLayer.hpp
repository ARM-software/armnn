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

    std::vector<TensorShape> InferOutputShapes(const std::vector<TensorShape>& inputShapes) const override;

    // Free up the constant source data
    void ReleaseConstantData() override {};

    std::unique_ptr<ScopedCpuTensorHandle> m_LayerOutput;
protected:
    ConstantLayer(const char* name);
    ~ConstantLayer() = default;

    ConstantTensors GetConstantTensorsByRef() override { return {m_LayerOutput}; }

};

} // namespace
