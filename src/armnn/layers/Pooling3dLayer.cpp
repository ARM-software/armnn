//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "Pooling3dLayer.hpp"

#include "LayerCloneBase.hpp"

#include <armnn/TypesUtils.hpp>

#include <armnnUtils/DataLayoutIndexed.hpp>

#include <armnn/backends/WorkloadData.hpp>
#include <armnn/backends/WorkloadFactory.hpp>

using namespace armnnUtils;

namespace armnn
{

Pooling3dLayer::Pooling3dLayer(const Pooling3dDescriptor& param, const char* name)
    : LayerWithParameters(1, 1, LayerType::Pooling3d, param, name)
{
}

std::unique_ptr<IWorkload> Pooling3dLayer::CreateWorkload(const IWorkloadFactory& factory) const
{
    Pooling3dQueueDescriptor descriptor;
    SetAdditionalInfo(descriptor);

    return factory.CreateWorkload(LayerType::Pooling3d, descriptor, PrepInfoAndDesc(descriptor));
}

Pooling3dLayer* Pooling3dLayer::Clone(Graph& graph) const
{
    return CloneBase<Pooling3dLayer>(graph, m_Param, GetName());
}

std::vector<TensorShape> Pooling3dLayer::InferOutputShapes(const std::vector<TensorShape>& inputShapes) const
{
    ARMNN_ASSERT(inputShapes.size() == 1);
    const TensorShape& inputShape = inputShapes[0];
    const DataLayoutIndexed dimensionIndices = m_Param.m_DataLayout;

    // If we support multiple batch dimensions in the future, then this assert will need to change.
    ARMNN_ASSERT_MSG(inputShape.GetNumDimensions() == 5, "Pooling3dLayer will always have 5D input.");

    unsigned int inWidth = inputShape[dimensionIndices.GetWidthIndex()];
    unsigned int inHeight = inputShape[dimensionIndices.GetHeightIndex()];
    unsigned int inDepth = inputShape[dimensionIndices.GetDepthIndex()];
    unsigned int inChannels = inputShape[dimensionIndices.GetChannelsIndex()];
    unsigned int inBatchSize = inputShape[0];

    bool isGlobalPooling = (m_Param.m_StrideX==0 && m_Param.m_StrideY==0 && m_Param.m_StrideZ==0);
    unsigned int outWidth = 1;
    unsigned int outHeight = 1;
    unsigned int outDepth = 1;
    if (!isGlobalPooling)
    {
        ARMNN_ASSERT_MSG(m_Param.m_StrideX!=0 && m_Param.m_StrideY!=0 && m_Param.m_StrideZ!=0,
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

                // Makes sure that border operations will start from inside the input and not the padded area.
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
        outDepth = CalcSize(inDepth, m_Param.m_PadFront, m_Param.m_PadBack, m_Param.m_PoolDepth, m_Param.m_StrideZ,
                            m_Param.m_OutputShapeRounding);
    }
    unsigned int outChannels = inChannels;
    unsigned int outBatchSize = inBatchSize;

    TensorShape tensorShape = m_Param.m_DataLayout == armnn::DataLayout::NDHWC ?
        TensorShape( { outBatchSize, outDepth, outHeight, outWidth, outChannels } ) :
        TensorShape( { outBatchSize, outChannels, outDepth, outHeight, outWidth });

    return std::vector<TensorShape>({ tensorShape });
}

void Pooling3dLayer::ValidateTensorShapesFromInputs()
{
    VerifyLayerConnections(1, CHECK_LOCATION());

    const TensorShape& outputShape = GetOutputSlot(0).GetTensorInfo().GetShape();

    VerifyShapeInferenceType(outputShape, m_ShapeInferenceMethod);

    auto inferredShapes = InferOutputShapes({ GetInputSlot(0).GetConnection()->GetTensorInfo().GetShape() });

    ARMNN_ASSERT(inferredShapes.size() == 1);

    ValidateAndCopyShape(outputShape, inferredShapes[0], m_ShapeInferenceMethod, "Pooling3dLayer");
}

ARMNN_NO_DEPRECATE_WARN_BEGIN
void Pooling3dLayer::Accept(ILayerVisitor& visitor) const
{
    visitor.VisitPooling3dLayer(this, GetParameters(), GetName());
}
ARMNN_NO_DEPRECATE_WARN_END

} // namespace armnn
