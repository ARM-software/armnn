//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "Pooling2dLayer.hpp"

#include "LayerCloneBase.hpp"

#include <armnn/TypesUtils.hpp>

#include <armnnUtils/DataLayoutIndexed.hpp>

#include <armnn/backends/WorkloadData.hpp>
#include <armnn/backends/WorkloadFactory.hpp>

using namespace armnnUtils;

namespace armnn
{

Pooling2dLayer::Pooling2dLayer(const Pooling2dDescriptor& param, const char* name)
    : LayerWithParameters(1, 1, LayerType::Pooling2d, param, name)
{
}

std::unique_ptr<IWorkload> Pooling2dLayer::CreateWorkload(const IWorkloadFactory& factory) const
{
    Pooling2dQueueDescriptor descriptor;
    SetAdditionalInfo(descriptor);

    return factory.CreateWorkload(LayerType::Pooling2d, descriptor, PrepInfoAndDesc(descriptor));
}

Pooling2dLayer* Pooling2dLayer::Clone(Graph& graph) const
{
    return CloneBase<Pooling2dLayer>(graph, m_Param, GetName());
}

std::vector<TensorShape> Pooling2dLayer::InferOutputShapes(const std::vector<TensorShape>& inputShapes) const
{
    ARMNN_ASSERT(inputShapes.size() == 1);
    const TensorShape& inputShape = inputShapes[0];
    const DataLayoutIndexed dimensionIndices = m_Param.m_DataLayout;

    // If we support multiple batch dimensions in the future, then this assert will need to change.
    ARMNN_ASSERT_MSG(inputShape.GetNumDimensions() == 4, "Pooling2dLayer will always have 4D input.");

    unsigned int inWidth = inputShape[dimensionIndices.GetWidthIndex()];
    unsigned int inHeight = inputShape[dimensionIndices.GetHeightIndex()];
    unsigned int inChannels = inputShape[dimensionIndices.GetChannelsIndex()];
    unsigned int inBatchSize = inputShape[0];

    bool isGlobalPooling = (m_Param.m_StrideX==0 && m_Param.m_StrideY==0);
    unsigned int outWidth = 1;
    unsigned int outHeight = 1;
    if (!isGlobalPooling)
    {
        ARMNN_ASSERT_MSG(m_Param.m_StrideX!=0 && m_Param.m_StrideY!=0,
                         "Stride can only be zero when performing global pooling");

        auto CalcSize = [](auto inSize, auto lowPad, auto highPad, auto poolSize, auto stride, auto outputShapeRounding)
            {
                unsigned int readSize = inSize + lowPad + highPad - poolSize;
                float div = static_cast<float>(readSize) / static_cast<float>(stride);

                unsigned int size = 0;
                switch (outputShapeRounding)
                {
                    case OutputShapeRounding::Ceiling:
                        size = static_cast<unsigned int>(ceil(div)) + 1;
                        break;
                    case OutputShapeRounding ::Floor:
                        size = static_cast<unsigned int>(floor(div)) + 1;
                        break;
                    default:
                        ARMNN_ASSERT_MSG(false, "Unsupported Output Shape Rounding");
                }

                // MakeS sure that border operations will start from inside the input and not the padded area.
                // This is what CL does...
                if ((size - 1)*stride >= inSize + lowPad)
                {
                    --size;
                }

                return size;
            };

        outWidth = CalcSize(inWidth, m_Param.m_PadLeft, m_Param.m_PadRight, m_Param.m_PoolWidth, m_Param.m_StrideX,
                            m_Param.m_OutputShapeRounding);
        outHeight = CalcSize(inHeight, m_Param.m_PadTop, m_Param.m_PadBottom, m_Param.m_PoolHeight, m_Param.m_StrideY,
                            m_Param.m_OutputShapeRounding);
    }
    unsigned int outChannels = inChannels;
    unsigned int outBatchSize = inBatchSize;

    TensorShape tensorShape = m_Param.m_DataLayout == armnn::DataLayout::NHWC ?
        TensorShape( { outBatchSize, outHeight, outWidth, outChannels } ) :
        TensorShape( { outBatchSize, outChannels, outHeight, outWidth });

    return std::vector<TensorShape>({ tensorShape });
}

void Pooling2dLayer::ValidateTensorShapesFromInputs()
{
    VerifyLayerConnections(1, CHECK_LOCATION());

    const TensorShape& outputShape = GetOutputSlot(0).GetTensorInfo().GetShape();

    VerifyShapeInferenceType(outputShape, m_ShapeInferenceMethod);

    auto inferredShapes = InferOutputShapes({ GetInputSlot(0).GetConnection()->GetTensorInfo().GetShape() });

    ARMNN_ASSERT(inferredShapes.size() == 1);

    ValidateAndCopyShape(outputShape, inferredShapes[0], m_ShapeInferenceMethod, "Pooling2dLayer");
}

void Pooling2dLayer::ExecuteStrategy(IStrategy& strategy) const
{
    strategy.ExecuteStrategy(this, GetParameters(), {}, GetName());
}

} // namespace armnn
