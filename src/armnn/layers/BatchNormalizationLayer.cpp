//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include "BatchNormalizationLayer.hpp"

#include "LayerCloneBase.hpp"

#include <armnn/TypesUtils.hpp>
#include <armnn/backends/TensorHandle.hpp>
#include <armnn/backends/WorkloadFactory.hpp>

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

    return factory.CreateWorkload(LayerType::BatchNormalization, descriptor, PrepInfoAndDesc(descriptor));
}

BatchNormalizationLayer* BatchNormalizationLayer::Clone(Graph& graph) const
{
    auto layer = CloneBase<BatchNormalizationLayer>(graph, m_Param, GetName());

    layer->m_Mean = m_Mean ? m_Mean : nullptr;
    layer->m_Variance = m_Variance ? m_Variance : nullptr;
    layer->m_Beta = m_Beta ? m_Beta : nullptr;
    layer->m_Gamma = m_Gamma ? m_Gamma : nullptr;

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
    // For API stability DO NOT ALTER order and add new members to the end of vector
    return {m_Mean, m_Variance, m_Beta, m_Gamma};
}

ARMNN_NO_DEPRECATE_WARN_BEGIN
void BatchNormalizationLayer::Accept(ILayerVisitor& visitor) const
{
    ManagedConstTensorHandle managedMean(m_Mean);
    ManagedConstTensorHandle managedVariance(m_Variance);
    ManagedConstTensorHandle managedBeta(m_Beta);
    ManagedConstTensorHandle managedGamma(m_Gamma);

    ConstTensor meanTensor(managedMean.GetTensorInfo(), managedMean.Map());
    ConstTensor varianceTensor(managedVariance.GetTensorInfo(), managedVariance.Map());
    ConstTensor betaTensor(managedBeta.GetTensorInfo(), managedBeta.Map());
    ConstTensor gammaTensor(managedGamma.GetTensorInfo(), managedGamma.Map());

    visitor.VisitBatchNormalizationLayer(
            this, GetParameters(), meanTensor, varianceTensor, betaTensor, gammaTensor, GetName());
}
ARMNN_NO_DEPRECATE_WARN_END

void BatchNormalizationLayer::ExecuteStrategy(IStrategy& strategy) const
{
    ManagedConstTensorHandle managedMean(m_Mean);
    ManagedConstTensorHandle managedVariance(m_Variance);
    ManagedConstTensorHandle managedBeta(m_Beta);
    ManagedConstTensorHandle managedGamma(m_Gamma);

    std::vector<armnn::ConstTensor> constTensors { { managedMean.GetTensorInfo(), managedMean.Map() },
                                                   { managedVariance.GetTensorInfo(), managedVariance.Map() },
                                                   { managedBeta.GetTensorInfo(), managedBeta.Map() },
                                                   { managedGamma.GetTensorInfo(), managedGamma.Map() } };

    strategy.ExecuteStrategy(this, GetParameters(), constTensors, GetName());
}

} // namespace armnn
