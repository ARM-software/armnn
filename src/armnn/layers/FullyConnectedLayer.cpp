//
// Copyright © 2017,2022 Arm Ltd and Contributors. All rights reserved.
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
    SetAdditionalInfo(descriptor);
    return factory.CreateWorkload(LayerType::FullyConnected, descriptor, PrepInfoAndDesc(descriptor));
}

FullyConnectedLayer* FullyConnectedLayer::Clone(Graph& graph) const
{
    auto layer = CloneBase<FullyConnectedLayer>(graph, m_Param, GetName());
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

Layer::ImmutableConstantTensors FullyConnectedLayer::GetConstantTensorsByRef() const
{
    Layer::ImmutableConstantTensors tensors = GetConnectedConstantAsInputTensors();
    return tensors;
}

void FullyConnectedLayer::ExecuteStrategy(IStrategy& strategy) const
{
    strategy.ExecuteStrategy(this, GetParameters(), {}, GetName());
}

} // namespace armnn
