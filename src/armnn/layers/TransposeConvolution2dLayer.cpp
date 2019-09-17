//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include "TransposeConvolution2dLayer.hpp"

#include "LayerCloneBase.hpp"

#include <armnn/TypesUtils.hpp>

#include <backendsCommon/CpuTensorHandle.hpp>
#include <backendsCommon/WorkloadFactory.hpp>

#include <DataLayoutIndexed.hpp>

using namespace armnnUtils;

namespace armnn
{

TransposeConvolution2dLayer::TransposeConvolution2dLayer(const TransposeConvolution2dDescriptor& param,
                                                         const char* name)
    : LayerWithParameters(1, 1, LayerType::TransposeConvolution2d, param, name)
{
}

std::unique_ptr<IWorkload> TransposeConvolution2dLayer::CreateWorkload(const Graph& graph,
                                                                       const IWorkloadFactory& factory) const
{
    BOOST_ASSERT_MSG(m_Weight != nullptr, "TransposeConvolution2dLayer: Weights data should not be null.");

    TransposeConvolution2dQueueDescriptor descriptor;
    descriptor.m_Weight = m_Weight.get();

    if (m_Param.m_BiasEnabled)
    {
        BOOST_ASSERT_MSG(m_Bias != nullptr, "TransposeConvolution2dLayer: Bias data should not be null.");
        descriptor.m_Bias = m_Bias.get();
    }

    return factory.CreateTransposeConvolution2d(descriptor, PrepInfoAndDesc(descriptor, graph));
}

TransposeConvolution2dLayer* TransposeConvolution2dLayer::Clone(Graph& graph) const
{
    auto layer = CloneBase<TransposeConvolution2dLayer>(graph, m_Param, GetName());

    layer->m_Weight = m_Weight ? std::make_unique<ScopedCpuTensorHandle>(*m_Weight) : nullptr;

    if (layer->m_Param.m_BiasEnabled)
    {
        layer->m_Bias = m_Bias ? std::make_unique<ScopedCpuTensorHandle>(*m_Bias) : nullptr;
    }

    return std::move(layer);
}

std::vector<TensorShape> TransposeConvolution2dLayer::InferOutputShapes(
    const std::vector<TensorShape>& inputShapes) const
{
    BOOST_ASSERT(inputShapes.size() == 2);
    const TensorShape& inputShape  = inputShapes[0];
    const TensorShape& kernelShape = inputShapes[1];

    BOOST_ASSERT_MSG(inputShape.GetNumDimensions() == 4, "Transpose convolutions will always have 4D input");

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

    unsigned int kernelElements = kernelShape[0] * kernelShape[dataLayoutIndex.GetChannelsIndex()];
    unsigned int inputElements  = batches * inputShape[dataLayoutIndex.GetChannelsIndex()];

    BOOST_ASSERT_MSG(inputElements != 0, "Invalid number of input elements");
    BOOST_ASSERT_MSG(kernelElements % inputElements == 0, "Invalid number of elements");

    unsigned int channels =  kernelElements / inputElements;

    TensorShape tensorShape = m_Param.m_DataLayout == armnn::DataLayout::NHWC ?
         TensorShape( { batches, hOutput, wOutput, channels } ) :
         TensorShape( { batches, channels, hOutput, wOutput });

    return std::vector<TensorShape>({ tensorShape });
}

void TransposeConvolution2dLayer::ValidateTensorShapesFromInputs()
{
    VerifyLayerConnections(1, CHECK_LOCATION());

    BOOST_ASSERT_MSG(m_Weight != nullptr, "TransposeConvolution2dLayer: Weight data cannot be null.");

    auto inferredShapes = InferOutputShapes({
         GetInputSlot(0).GetConnection()->GetTensorInfo().GetShape(),
         m_Weight->GetTensorInfo().GetShape() });

    BOOST_ASSERT(inferredShapes.size() == 1);

    ConditionalThrowIfNotEqual<LayerValidationException>(
        "TransposeConvolution2dLayer: TensorShape set on OutputSlot[0] does not match the inferred shape.",
        GetOutputSlot(0).GetTensorInfo().GetShape(),
        inferredShapes[0]);
}

Layer::ConstantTensors TransposeConvolution2dLayer::GetConstantTensorsByRef()
{
    return {m_Weight, m_Bias};
}

void TransposeConvolution2dLayer::Accept(ILayerVisitor& visitor) const
{
    ConstTensor weightsTensor(m_Weight->GetTensorInfo(), m_Weight->Map(true)) ;
    Optional<ConstTensor> optionalBiasTensor = EmptyOptional();

    if (GetParameters().m_BiasEnabled)
    {
        ConstTensor biasTensor(m_Bias->GetTensorInfo(), m_Bias->Map(true));
        optionalBiasTensor = Optional<ConstTensor>(biasTensor);
    }

    visitor.VisitTransposeConvolution2dLayer(this, GetParameters(), weightsTensor, optionalBiasTensor, GetName());
}

} // namespace armnn
