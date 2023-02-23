//
// Copyright © 2017-2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "DetectionPostProcessLayer.hpp"

#include "LayerCloneBase.hpp"

#include <armnn/TypesUtils.hpp>
#include <armnn/backends/TensorHandle.hpp>
#include <armnn/backends/WorkloadData.hpp>
#include <armnn/backends/WorkloadFactory.hpp>

namespace armnn
{

DetectionPostProcessLayer::DetectionPostProcessLayer(const DetectionPostProcessDescriptor& param, const char* name)
    : LayerWithParameters(2, 4, LayerType::DetectionPostProcess, param, name)
{
}

std::unique_ptr<IWorkload> DetectionPostProcessLayer::CreateWorkload(const armnn::IWorkloadFactory& factory) const
{
    DetectionPostProcessQueueDescriptor descriptor;
    descriptor.m_Anchors = m_Anchors.get();
    SetAdditionalInfo(descriptor);

    return factory.CreateWorkload(LayerType::DetectionPostProcess, descriptor, PrepInfoAndDesc(descriptor));
}

DetectionPostProcessLayer* DetectionPostProcessLayer::Clone(Graph& graph) const
{
    auto layer = CloneBase<DetectionPostProcessLayer>(graph, m_Param, GetName());
    layer->m_Anchors = m_Anchors ? m_Anchors : nullptr;
    return std::move(layer);
}

void DetectionPostProcessLayer::ValidateTensorShapesFromInputs()
{
    VerifyLayerConnections(2, CHECK_LOCATION());

    const TensorShape& outputShape = GetOutputSlot(0).GetTensorInfo().GetShape();

    VerifyShapeInferenceType(outputShape, m_ShapeInferenceMethod);

    // on this level constant data should not be released.
    ARMNN_ASSERT_MSG(m_Anchors != nullptr, "DetectionPostProcessLayer: Anchors data should not be null.");

    ARMNN_ASSERT_MSG(GetNumOutputSlots() == 4, "DetectionPostProcessLayer: The layer should return 4 outputs.");

    std::vector<TensorShape> inferredShapes = InferOutputShapes(
            { GetInputSlot(0).GetConnection()->GetTensorInfo().GetShape(),
              GetInputSlot(1).GetConnection()->GetTensorInfo().GetShape() });

    ARMNN_ASSERT(inferredShapes.size() == 4);
    ARMNN_ASSERT(inferredShapes[0].GetDimensionality() == Dimensionality::Specified);
    ARMNN_ASSERT(inferredShapes[1].GetDimensionality() == Dimensionality::Specified);
    ARMNN_ASSERT(inferredShapes[2].GetDimensionality() == Dimensionality::Specified);
    ARMNN_ASSERT(inferredShapes[3].GetDimensionality() == Dimensionality::Specified);

    ValidateAndCopyShape(outputShape, inferredShapes[0], m_ShapeInferenceMethod, "DetectionPostProcessLayer");

    ValidateAndCopyShape(GetOutputSlot(1).GetTensorInfo().GetShape(),
                         inferredShapes[1],
                         m_ShapeInferenceMethod,
                         "DetectionPostProcessLayer", 1);

    ValidateAndCopyShape(GetOutputSlot(2).GetTensorInfo().GetShape(),
                         inferredShapes[2],
                         m_ShapeInferenceMethod,
                         "DetectionPostProcessLayer", 2);

    ValidateAndCopyShape(GetOutputSlot(3).GetTensorInfo().GetShape(),
                         inferredShapes[3],
                         m_ShapeInferenceMethod,
                         "DetectionPostProcessLayer", 3);
}

std::vector<TensorShape> DetectionPostProcessLayer::InferOutputShapes(const std::vector<TensorShape>&) const
{
    unsigned int detectedBoxes = m_Param.m_MaxDetections * m_Param.m_MaxClassesPerDetection;

    std::vector<TensorShape> results;
    results.push_back({ 1, detectedBoxes, 4 });
    results.push_back({ 1, detectedBoxes });
    results.push_back({ 1, detectedBoxes });
    results.push_back({ 1 });
    return results;
}

Layer::ImmutableConstantTensors DetectionPostProcessLayer::GetConstantTensorsByRef() const
{
    // For API stability DO NOT ALTER order and add new members to the end of vector
    return { m_Anchors };
}

void DetectionPostProcessLayer::ExecuteStrategy(IStrategy& strategy) const
{
    ManagedConstTensorHandle managedAnchors(m_Anchors);
    std::vector<armnn::ConstTensor> constTensors { {managedAnchors.GetTensorInfo(), managedAnchors.Map()} };
    strategy.ExecuteStrategy(this, GetParameters(), constTensors, GetName());
}

} // namespace armnn
