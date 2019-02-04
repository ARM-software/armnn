//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "DetectionPostProcessLayer.hpp"

#include "LayerCloneBase.hpp"

#include <armnn/TypesUtils.hpp>
#include <backendsCommon/CpuTensorHandle.hpp>
#include <backendsCommon/WorkloadData.hpp>
#include <backendsCommon/WorkloadFactory.hpp>

namespace armnn
{

DetectionPostProcessLayer::DetectionPostProcessLayer(const DetectionPostProcessDescriptor& param, const char* name)
    : LayerWithParameters(2, 4, LayerType::DetectionPostProcess, param, name)
{
}

std::unique_ptr<IWorkload> DetectionPostProcessLayer::CreateWorkload(const armnn::Graph& graph,
                                                                     const armnn::IWorkloadFactory& factory) const
{
    DetectionPostProcessQueueDescriptor descriptor;
    descriptor.m_Anchors = m_Anchors.get();
    return factory.CreateDetectionPostProcess(descriptor, PrepInfoAndDesc(descriptor, graph));
}

DetectionPostProcessLayer* DetectionPostProcessLayer::Clone(Graph& graph) const
{
    auto layer = CloneBase<DetectionPostProcessLayer>(graph, m_Param, GetName());
    layer->m_Anchors = m_Anchors ? std::make_unique<ScopedCpuTensorHandle>(*m_Anchors) : nullptr;
    return std::move(layer);
}

void DetectionPostProcessLayer::ValidateTensorShapesFromInputs()
{
    VerifyLayerConnections(2, CHECK_LOCATION());

    // on this level constant data should not be released.
    BOOST_ASSERT_MSG(m_Anchors != nullptr, "DetectionPostProcessLayer: Anchors data should not be null.");

    BOOST_ASSERT_MSG(GetNumOutputSlots() == 4, "DetectionPostProcessLayer: The layer should return 4 outputs.");

    unsigned int detectedBoxes = m_Param.m_MaxDetections * m_Param.m_MaxClassesPerDetection;

    const TensorShape& inferredDetectionBoxes = TensorShape({ 1, detectedBoxes, 4 });
    const TensorShape& inferredDetectionScores = TensorShape({ 1, detectedBoxes });
    const TensorShape& inferredNumberDetections = TensorShape({ 1 });

    ConditionalThrowIfNotEqual<LayerValidationException>(
        "DetectionPostProcessLayer: TensorShape set on OutputSlot[0] does not match the inferred shape.",
        GetOutputSlot(0).GetTensorInfo().GetShape(),
        inferredDetectionBoxes);
    ConditionalThrowIfNotEqual<LayerValidationException>(
        "DetectionPostProcessLayer: TensorShape set on OutputSlot[1] does not match the inferred shape.",
        GetOutputSlot(1).GetTensorInfo().GetShape(),
        inferredDetectionScores);
    ConditionalThrowIfNotEqual<LayerValidationException>(
        "DetectionPostProcessLayer: TensorShape set on OutputSlot[2] does not match the inferred shape.",
        GetOutputSlot(2).GetTensorInfo().GetShape(),
        inferredDetectionScores);
    ConditionalThrowIfNotEqual<LayerValidationException>(
        "DetectionPostProcessLayer: TensorShape set on OutputSlot[3] does not match the inferred shape.",
        GetOutputSlot(3).GetTensorInfo().GetShape(),
        inferredNumberDetections);
}

Layer::ConstantTensors DetectionPostProcessLayer::GetConstantTensorsByRef()
{
    return { m_Anchors };
}

void DetectionPostProcessLayer::Accept(ILayerVisitor& visitor) const
{
    ConstTensor anchorTensor(m_Anchors->GetTensorInfo(), m_Anchors->GetConstTensor<void>());
    visitor.VisitDetectionPostProcessLayer(this, GetParameters(), anchorTensor, GetName());
}

} // namespace armnn
