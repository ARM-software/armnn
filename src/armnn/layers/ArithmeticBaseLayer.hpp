//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//
#pragma once

#include <Layer.hpp>

namespace armnn
{

/// NOTE: this is an abstract class, it does not implement:
///  std::unique_ptr<IWorkload> Layer::CreateWorkload(const Graph& graph, const IWorkloadFactory& factory) const = 0;
///  Layer* Clone(Graph& graph) const = 0;
class ArithmeticBaseLayer : public Layer
{
public:
    void ValidateTensorShapesFromInputs() override;
    std::vector<TensorShape> InferOutputShapes(const std::vector<TensorShape>& inputShapes) const override;

protected:
    ArithmeticBaseLayer(unsigned int numInputSlots, unsigned int numOutputSlots, LayerType type, const char* name);
    ~ArithmeticBaseLayer() = default;
};

} // namespace
