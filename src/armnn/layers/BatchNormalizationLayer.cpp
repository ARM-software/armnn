//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include "BatchNormalizationLayer.hpp"

#include "LayerCloneBase.hpp"

#include <armnn/TypesUtils.hpp>
#include <backendsCommon/CpuTensorHandle.hpp>
#include <backendsCommon/WorkloadFactory.hpp>

namespace armnn
{

BatchNormalizationLayer::BatchNormalizationLayer(const armnn::BatchNormalizationDescriptor& param, const char* name)
    : LayerWithParameters(1, 1, LayerType::BatchNormalization, param, name)
{
}

std::unique_ptr<IWorkload> BatchNormalizationLayer::CreateWorkload(const IWorkloadFactory& factory) const
{
    // on this level constant data should not be released..
    ARMNN_ASSERT_MSG(m_Mean != nullptr, "BatchNormalizationLayer: Mean data should not be null.");
    ARMNN_ASSERT_MSG(m_Variance != nullptr, "BatchNormalizationLayer: Variance data should not be null.");
    ARMNN_ASSERT_MSG(m_Beta != nullptr, "BatchNormalizationLayer: Beta data should not be null.");
    ARMNN_ASSERT_MSG(m_Gamma != nullptr, "BatchNormalizationLayer: Gamma data should not be null.");

    BatchNormalizationQueueDescriptor descriptor;
    SetAdditionalInfo(descriptor);

    descriptor.m_Mean = m_Mean.get();
    descriptor.m_Variance = m_Variance.get();
    descriptor.m_Beta = m_Beta.get();
    descriptor.m_Gamma = m_Gamma.get();

    return factory.CreateBatchNormalization(descriptor, PrepInfoAndDesc(descriptor));
}

BatchNormalizationLayer* BatchNormalizationLayer::Clone(Graph& graph) const
{
    auto layer = CloneBase<BatchNormalizationLayer>(graph, m_Param, GetName());

    layer->m_Mean = m_Mean ? std::make_unique<ScopedCpuTensorHandle>(*m_Mean) : nullptr;
    layer->m_Variance = m_Variance ? std::make_unique<ScopedCpuTensorHandle>(*m_Variance) : nullptr;
    layer->m_Beta = m_Beta ? std::make_unique<ScopedCpuTensorHandle>(*m_Beta) : nullptr;
    layer->m_Gamma = m_Gamma ? std::make_unique<ScopedCpuTensorHandle>(*m_Gamma) : nullptr;

    return std::move(layer);
}

void BatchNormalizationLayer::ValidateTensorShapesFromInputs()
{
    VerifyLayerConnections(1, CHECK_LOCATION());

    const TensorShape& outputShape = GetOutputSlot(0).GetTensorInfo().GetShape();

    VerifyShapeInferenceType(outputShape, m_ShapeInferenceMethod);

    auto inferredShapes = InferOutputShapes({ GetInputSlot(0).GetConnection()->GetTensorInfo().GetShape() });

    ARMNN_ASSERT(inferredShapes.size() == 1);

    ValidateAndCopyShape(outputShape, inferredShapes[0], m_ShapeInferenceMethod, "BatchNormalizationLayer");

}

Layer::ConstantTensors BatchNormalizationLayer::GetConstantTensorsByRef()
{
    return {m_Mean, m_Variance, m_Beta, m_Gamma};
}

void BatchNormalizationLayer::Accept(ILayerVisitor& visitor) const
{
    ConstTensor meanTensor(m_Mean->GetTensorInfo(), m_Mean->Map(true));
    ConstTensor varianceTensor(m_Variance->GetTensorInfo(), m_Variance->Map(true));
    ConstTensor betaTensor(m_Beta->GetTensorInfo(), m_Beta->Map(true));
    ConstTensor gammaTensor(m_Gamma->GetTensorInfo(), m_Gamma->Map(true));
    visitor.VisitBatchNormalizationLayer(
            this, GetParameters(), meanTensor, varianceTensor, betaTensor, gammaTensor, GetName());
}

} // namespace armnn
