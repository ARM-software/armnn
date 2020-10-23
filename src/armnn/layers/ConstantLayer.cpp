//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include "ConstantLayer.hpp"
#include "LayerCloneBase.hpp"

#include <armnn/TypesUtils.hpp>
#include <backendsCommon/CpuTensorHandle.hpp>
#include <backendsCommon/WorkloadData.hpp>
#include <backendsCommon/WorkloadFactory.hpp>

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

    return factory.CreateConstant(descriptor, PrepInfoAndDesc(descriptor));
}

ConstantLayer* ConstantLayer::Clone(Graph& graph) const
{
    // Cloned layers share the same layer output object.
    auto layer = CloneBase<ConstantLayer>(graph, GetName());

    layer->m_LayerOutput = m_LayerOutput ? std::make_unique<ScopedCpuTensorHandle>(*m_LayerOutput) : nullptr;

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

void ConstantLayer::Accept(ILayerVisitor& visitor) const
{
    ConstTensor layerOutputTensor(m_LayerOutput->GetTensorInfo(), m_LayerOutput->Map(true)) ;
    visitor.VisitConstantLayer(this, layerOutputTensor, GetName());
}

} // namespace armnn
