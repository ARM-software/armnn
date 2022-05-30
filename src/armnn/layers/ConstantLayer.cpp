//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include "ConstantLayer.hpp"
#include "LayerCloneBase.hpp"

#include <armnn/TypesUtils.hpp>
#include <armnn/backends/TensorHandle.hpp>
#include <armnn/backends/WorkloadData.hpp>
#include <armnn/backends/WorkloadFactory.hpp>

namespace armnn
{

ConstantLayer::ConstantLayer(const char* name)
    : Layer(0, 1, LayerType::Constant, name)
{
}

std::unique_ptr<IWorkload> ConstantLayer::CreateWorkload(const IWorkloadFactory& factory) const
{
    ConstantQueueDescriptor descriptor;
    descriptor.m_LayerOutput = m_LayerOutput.get();
    SetAdditionalInfo(descriptor);

    return factory.CreateWorkload(LayerType::Constant, descriptor, PrepInfoAndDesc(descriptor));
}

ConstantLayer* ConstantLayer::Clone(Graph& graph) const
{
    // Cloned layers share the same layer output object.
    auto layer = CloneBase<ConstantLayer>(graph, GetName());

    layer->m_LayerOutput = m_LayerOutput ? m_LayerOutput : nullptr;

    return std::move(layer);
}

std::vector<TensorShape> ConstantLayer::InferOutputShapes(const std::vector<TensorShape>& inputShapes) const
{
    return std::vector<TensorShape>({  inputShapes[0] });
}

void ConstantLayer::ValidateTensorShapesFromInputs()
{

    // Get the output shape from the value of the constant layer.
    TensorShape const& outShape = m_LayerOutput->GetTensorInfo().GetShape();

    ConditionalThrow<LayerValidationException>(
            outShape.GetDimensionality() != Dimensionality::NotSpecified,
            "Constant layer m_LayerOutput output shape can not be Dimensionality::NotSpecified");

    ConditionalThrow<LayerValidationException>(
            outShape.AreAllDimensionsSpecified(),
            "Constant layer m_LayerOutput output shape can not have an unspecified dimension");

    ConditionalThrowIfNotEqual<LayerValidationException>(
               "ConstantLayer: TensorShape set on OutputSlot[0] does not match the inferred shape.",
               GetOutputSlot(0).GetTensorInfo().GetShape(),
               outShape);
}

void ConstantLayer::ExecuteStrategy(IStrategy& strategy) const
{
    ManagedConstTensorHandle managedLayerOutput(m_LayerOutput);
    ConstTensor layerOutputTensor(managedLayerOutput.GetTensorInfo(), managedLayerOutput.Map());
    strategy.ExecuteStrategy(this, BaseDescriptor(), { layerOutputTensor }, GetName());
}

} // namespace armnn
