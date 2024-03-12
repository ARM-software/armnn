//
// Copyright © 2019-2024 Arm Ltd and Contributors. All rights reserved.
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
    if (!m_Weight)
    {
        throw armnn::NullPointerException("TransposeConvolution2dLayer: Weights data should not be null.");
    }

    TransposeConvolution2dQueueDescriptor descriptor;
    descriptor.m_Weight = m_Weight.get();

    if (m_Param.m_BiasEnabled)
    {
        if (!m_Bias)
        {
            throw armnn::NullPointerException("TransposeConvolution2dLayer: Bias data should not be null.");
        }
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
    if (inputShapes.size() != 2)
    {
        throw armnn::Exception("inputShapes' size is \"" + std::to_string(inputShapes.size()) +
                               "\" - should be \"2\".");
    }

    const TensorShape& inputShape  = inputShapes[0];
    const TensorShape& kernelShape = inputShapes[1];

    if (inputShape.GetNumDimensions() != 4)
    {
        throw armnn::Exception("Transpose convolutions will always have 4D input");
    }

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

    if (!m_Weight)
    {
        throw armnn::LayerValidationException("TransposeConvolution2dLayer: Weight data cannot be null.");
    }

    std::vector<TensorShape> expectedOutputShape;
    std::vector<TensorShape> outputShapeGivenAsInput;

    expectedOutputShape = InferOutputShapes({GetInputSlot(0).GetTensorInfo().GetShape(),
                                             m_Weight->GetTensorInfo().GetShape() });

    if (expectedOutputShape.size() != 1)
    {
        throw armnn::LayerValidationException("expectedOutputShape' size is "
                                              + std::to_string(expectedOutputShape.size()) +
                                              " - should be \"1\".");
    }

    // If output_shape was specified then use it rather than calculate an inferred output shape.
    if (m_Param.m_OutputShapeEnabled)
    {
        TensorShape shapeAsTensorShape(static_cast<unsigned int>(m_Param.m_OutputShape.size()),
            m_Param.m_OutputShape.data());
        outputShapeGivenAsInput.push_back(shapeAsTensorShape);

        if (outputShapeGivenAsInput.size() != 1)
        {
            throw armnn::LayerValidationException("outputShapeGivenAsInput' size is "
                                                  + std::to_string(outputShapeGivenAsInput.size()) +
                                                  " - should be \"1\".");
        }

        if (expectedOutputShape != outputShapeGivenAsInput)
        {
            throw armnn::LayerValidationException("TransposeConvolution2dLayer: "
                                                  "output calculated by InferOutputShapes and the output given "
                                                  "as an input parameter to the layer are not matching");
        }
    }

    ValidateAndCopyShape(outputShape, expectedOutputShape[0], m_ShapeInferenceMethod, "TransposeConvolution2dLayer");
}

Layer::ImmutableConstantTensors TransposeConvolution2dLayer::GetConstantTensorsByRef() const
{
    // For API stability DO NOT ALTER order and add new members to the end of vector
    return {m_Weight, m_Bias};
}

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
