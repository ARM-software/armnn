//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "TransposeConvolution2dLayer.hpp"
#include "LayerCloneBase.hpp"

#include <armnnUtils/DataLayoutIndexed.hpp>

#include <armnn/backends/TensorHandle.hpp>
#include <armnn/backends/WorkloadFactory.hpp>

using namespace armnnUtils;

namespace armnn
{

TransposeConvolution2dLayer::TransposeConvolution2dLayer(const TransposeConvolution2dDescriptor& param,
                                                         const char* name)
    : LayerWithParameters(1, 1, LayerType::TransposeConvolution2d, param, name)
{
}

std::unique_ptr<IWorkload> TransposeConvolution2dLayer::CreateWorkload(const IWorkloadFactory& factory) const
{
    ARMNN_ASSERT_MSG(m_Weight != nullptr, "TransposeConvolution2dLayer: Weights data should not be null.");

    TransposeConvolution2dQueueDescriptor descriptor;
    descriptor.m_Weight = m_Weight.get();

    if (m_Param.m_BiasEnabled)
    {
        ARMNN_ASSERT_MSG(m_Bias != nullptr, "TransposeConvolution2dLayer: Bias data should not be null.");
        descriptor.m_Bias = m_Bias.get();
    }

    SetAdditionalInfo(descriptor);

    return factory.CreateWorkload(LayerType::TransposeConvolution2d, descriptor, PrepInfoAndDesc(descriptor));
}

TransposeConvolution2dLayer* TransposeConvolution2dLayer::Clone(Graph& graph) const
{
    auto layer = CloneBase<TransposeConvolution2dLayer>(graph, m_Param, GetName());

    layer->m_Weight = m_Weight ? m_Weight : nullptr;

    if (layer->m_Param.m_BiasEnabled)
    {
        layer->m_Bias = m_Bias ? m_Bias : nullptr;
    }

    return std::move(layer);
}

std::vector<TensorShape> TransposeConvolution2dLayer::InferOutputShapes(
    const std::vector<TensorShape>& inputShapes) const
{
    ARMNN_ASSERT(inputShapes.size() == 2);
    const TensorShape& inputShape  = inputShapes[0];
    const TensorShape& kernelShape = inputShapes[1];

    ARMNN_ASSERT_MSG(inputShape.GetNumDimensions() == 4, "Transpose convolutions will always have 4D input");

    DataLayoutIndexed dataLayoutIndex(m_Param.m_DataLayout);

    const unsigned int batches = inputShape[0];

    const unsigned int wInput = inputShape[dataLayoutIndex.GetWidthIndex()];
    const unsigned int hInput = inputShape[dataLayoutIndex.GetHeightIndex()];

    const unsigned int wKernel = kernelShape[dataLayoutIndex.GetWidthIndex()];
    const unsigned int hKernel = kernelShape[dataLayoutIndex.GetHeightIndex()];

    unsigned int wPadding = m_Param.m_PadLeft + m_Param.m_PadRight;
    unsigned int hPadding = m_Param.m_PadTop + m_Param.m_PadBottom;

    unsigned int wOutput = (wInput - 1) * m_Param.m_StrideX + wKernel - wPadding;
    unsigned int hOutput = (hInput - 1) * m_Param.m_StrideY + hKernel - hPadding;
    unsigned int cOutput = kernelShape[0];

    TensorShape tensorShape = m_Param.m_DataLayout == armnn::DataLayout::NHWC ?
         TensorShape( { batches, hOutput, wOutput, cOutput } ) :
         TensorShape( { batches, cOutput, hOutput, wOutput });

    return std::vector<TensorShape>({ tensorShape });
}

void TransposeConvolution2dLayer::ValidateTensorShapesFromInputs()
{
    VerifyLayerConnections(1, CHECK_LOCATION());

    const TensorShape& outputShape = GetOutputSlot(0).GetTensorInfo().GetShape();

    VerifyShapeInferenceType(outputShape, m_ShapeInferenceMethod);

    ARMNN_ASSERT_MSG(m_Weight != nullptr, "TransposeConvolution2dLayer: Weight data cannot be null.");

    std::vector<TensorShape> expectedOutputShape;
    // If output_shape was specified then use it rather than calculate an inferred output shape.
    if (m_Param.m_OutputShapeEnabled)
    {
        TensorShape shapeAsTensorShape(static_cast<unsigned int>(m_Param.m_OutputShape.size()),
            m_Param.m_OutputShape.data());
        expectedOutputShape.push_back(shapeAsTensorShape);
    }
    else
    {
        expectedOutputShape = InferOutputShapes({GetInputSlot(0).GetConnection()->GetTensorInfo().GetShape(),
                                                 m_Weight->GetTensorInfo().GetShape() });
    }

    ARMNN_ASSERT(expectedOutputShape.size() == 1);

    ValidateAndCopyShape(outputShape, expectedOutputShape[0], m_ShapeInferenceMethod, "TransposeConvolution2dLayer");
}

Layer::ConstantTensors TransposeConvolution2dLayer::GetConstantTensorsByRef()
{
    // For API stability DO NOT ALTER order and add new members to the end of vector
    return {m_Weight, m_Bias};
}

ARMNN_NO_DEPRECATE_WARN_BEGIN
void TransposeConvolution2dLayer::Accept(ILayerVisitor& visitor) const
{
    ManagedConstTensorHandle managedWeight(m_Weight);
    ConstTensor weightsTensor(managedWeight.GetTensorInfo(), managedWeight.Map());

    Optional<ConstTensor> optionalBiasTensor = EmptyOptional();
    ManagedConstTensorHandle managedBias(m_Bias);
    if (GetParameters().m_BiasEnabled)
    {
        ConstTensor biasTensor(managedBias.GetTensorInfo(), managedBias.Map());
        optionalBiasTensor = Optional<ConstTensor>(biasTensor);
    }

    visitor.VisitTransposeConvolution2dLayer(this, GetParameters(), weightsTensor, optionalBiasTensor, GetName());
}
ARMNN_NO_DEPRECATE_WARN_END

void TransposeConvolution2dLayer::ExecuteStrategy(IStrategy& strategy) const
{
    ManagedConstTensorHandle managedWeight(m_Weight);
    std::vector<armnn::ConstTensor> constTensors { { managedWeight.GetTensorInfo(), managedWeight.Map() } };

    ManagedConstTensorHandle managedBias(m_Bias);
    if (GetParameters().m_BiasEnabled)
    {
        constTensors.emplace_back(ConstTensor(managedBias.GetTensorInfo(), managedBias.Map()));
    }

    strategy.ExecuteStrategy(this, GetParameters(), constTensors, GetName());
}

} // namespace armnn
