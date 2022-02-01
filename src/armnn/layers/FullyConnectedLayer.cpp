//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include "FullyConnectedLayer.hpp"

#include "LayerCloneBase.hpp"

#include <armnn/TypesUtils.hpp>
#include <armnn/backends/TensorHandle.hpp>
#include <armnn/backends/WorkloadData.hpp>
#include <armnn/backends/WorkloadFactory.hpp>

namespace armnn
{

FullyConnectedLayer::FullyConnectedLayer(const FullyConnectedDescriptor& param, const char* name)
    : LayerWithParameters(param.GetNumInputs(), 1, LayerType::FullyConnected, param, name)
{
}

std::unique_ptr<IWorkload> FullyConnectedLayer::CreateWorkload(const IWorkloadFactory& factory) const
{
    FullyConnectedQueueDescriptor descriptor;
    if (m_Weight)
    {
        descriptor.m_Weight = m_Weight.get();
    }
    if (m_Param.m_BiasEnabled && m_Bias)
    {
        descriptor.m_Bias = m_Bias.get();
    }
    SetAdditionalInfo(descriptor);

    return factory.CreateWorkload(LayerType::FullyConnected, descriptor, PrepInfoAndDesc(descriptor));
}

FullyConnectedLayer* FullyConnectedLayer::Clone(Graph& graph) const
{
    auto layer = CloneBase<FullyConnectedLayer>(graph, m_Param, GetName());
    layer->m_Weight = m_Weight ? m_Weight : nullptr;
    if (layer->m_Param.m_BiasEnabled)
    {
        layer->m_Bias = m_Bias ? m_Bias : nullptr;
    }
    return std::move(layer);
}

std::vector<TensorShape> FullyConnectedLayer::InferOutputShapes(const std::vector<TensorShape>& inputShapes) const
{
    ARMNN_ASSERT(inputShapes.size() == 2);
    const TensorShape& inputShape = inputShapes[0];
    const TensorShape weightShape = inputShapes[1];

    // Output for FC is [1, w[1]].
    unsigned int batches = inputShape[0];
    unsigned int dimIdx = m_Param.m_TransposeWeightMatrix ? 0 : 1;

    return std::vector<TensorShape>({ TensorShape({batches, weightShape[dimIdx]})});
}

void FullyConnectedLayer::ValidateTensorShapesFromInputs()
{
    const TensorShape& outputShape = GetOutputSlot(0).GetTensorInfo().GetShape();

    VerifyShapeInferenceType(outputShape, m_ShapeInferenceMethod);

    std::vector<TensorShape> inferredShapes = InferOutputShapes(
            {GetInputSlot(0).GetConnection()->GetTensorInfo().GetShape(),
             GetInputSlot(1).GetConnection()->GetTensorInfo().GetShape()});

    ARMNN_ASSERT(inferredShapes.size() == 1);
    ARMNN_ASSERT(inferredShapes[0].GetDimensionality() == Dimensionality::Specified);

    ValidateAndCopyShape(outputShape, inferredShapes[0], m_ShapeInferenceMethod, "FullyConnectedLayer");
}

Layer::ConstantTensors FullyConnectedLayer::GetConstantTensorsByRef()
{
    // For API stability DO NOT ALTER order and add new members to the end of vector
    return {m_Weight, m_Bias};
}

ARMNN_NO_DEPRECATE_WARN_BEGIN
void FullyConnectedLayer::Accept(ILayerVisitor& visitor) const
{
    visitor.VisitFullyConnectedLayer(this, GetParameters(), GetName());
}
ARMNN_NO_DEPRECATE_WARN_END

void FullyConnectedLayer::ExecuteStrategy(IStrategy& strategy) const
{
    strategy.ExecuteStrategy(this, GetParameters(), {}, GetName());
}

} // namespace armnn
