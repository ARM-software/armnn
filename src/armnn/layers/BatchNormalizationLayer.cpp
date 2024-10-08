//
// Copyright © 2017-2024 Arm Ltd and Contributors. All rights reserved.
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
    if (!m_Mean)
    {
        throw armnn::NullPointerException("BatchNormalizationLayer: Mean data should not be null.");
    }

    if (!m_Variance)
    {
        throw armnn::NullPointerException("BatchNormalizationLayer: Variance data should not be null.");
    }

    if (!m_Beta)
    {
        throw armnn::NullPointerException("BatchNormalizationLayer: Beta data should not be null.");
    }

    if (!m_Gamma)
    {
        throw armnn::NullPointerException("BatchNormalizationLayer: Gamma data should not be null.");
    }

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

    auto inferredShapes = InferOutputShapes({ GetInputSlot(0).GetTensorInfo().GetShape() });

    if (inferredShapes.size() != 1)
    {
        throw armnn::LayerValidationException("inferredShapes has "
                                              + std::to_string(inferredShapes.size()) +
                                              " elements - should only have 1.");
    }

    ValidateAndCopyShape(outputShape, inferredShapes[0], m_ShapeInferenceMethod, "BatchNormalizationLayer");

}

Layer::ImmutableConstantTensors BatchNormalizationLayer::GetConstantTensorsByRef() const
{
    // For API stability DO NOT ALTER order and add new members to the end of vector
    return {m_Mean, m_Variance, m_Beta, m_Gamma};
}

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
