//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//
#include "Pooling2dLayer.hpp"

#include "LayerCloneBase.hpp"

#include <armnn/TypesUtils.hpp>
#include <backends/WorkloadData.hpp>
#include <backends/WorkloadFactory.hpp>

namespace armnn
{

Pooling2dLayer::Pooling2dLayer(const Pooling2dDescriptor& param, const char* name)
    : LayerWithParameters(1, 1, LayerType::Pooling2d, param, name)
{
}

std::unique_ptr<IWorkload> Pooling2dLayer::CreateWorkload(const Graph& graph, const IWorkloadFactory& factory) const
{
    Pooling2dQueueDescriptor descriptor;
    return factory.CreatePooling2d(descriptor, PrepInfoAndDesc(descriptor, graph));
}

Pooling2dLayer* Pooling2dLayer::Clone(Graph& graph) const
{
    return CloneBase<Pooling2dLayer>(graph, m_Param, GetName());
}

void Pooling2dLayer::ValidateTensorShapesFromInputs()
{
    ConditionalThrow<LayerValidationException>(GetInputSlot(0).GetConnection() != nullptr,
                     "Pooling2dLayer: InputSlot must be connected to an OutputSlot");
    ConditionalThrow<LayerValidationException>(GetInputSlot(0).GetConnection()->IsTensorInfoSet(),
                     "Pooling2dLayer: TensorInfo must be set on connected InputSlot.");

    IOutputSlot* input = GetInputSlot(0).GetConnection();
    const TensorShape& inputShape = input->GetTensorInfo().GetShape();

    // If we support multiple batch dimensions in the future, then this assert will need to change.
    BOOST_ASSERT_MSG(inputShape.GetNumDimensions() == 4, "Pooling2dLayer will always have 4D input.");


    unsigned int inWidth = inputShape[3];
    unsigned int inHeight = inputShape[2];
    unsigned int inChannels = inputShape[1];
    unsigned int inBatchSize = inputShape[0];

    bool isGlobalPooling = (m_Param.m_StrideX==0 && m_Param.m_StrideY==0);
    unsigned int outWidth = 1;
    unsigned int outHeight = 1;
    if (!isGlobalPooling)
    {
        BOOST_ASSERT_MSG(m_Param.m_StrideX!=0 && m_Param.m_StrideY!=0,
                         "Stride can only be zero when performing global pooling");

        auto CalcSize = [](auto inSize, auto lowPad, auto highPad, auto poolSize, auto stride, auto padMethod,
                           auto outputShapeRounding)
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
                        BOOST_ASSERT_MSG(false, "Unsupported Output Shape Rounding");
                }

                // Make sure that border operations will start from inside the input and not the padded area
                // This is what both Caffe and CL does...
                if ((size - 1)*stride >= inSize + lowPad)
                {
                    --size;
                }

                return size;
            };

        outWidth = CalcSize(inWidth, m_Param.m_PadLeft, m_Param.m_PadRight, m_Param.m_PoolWidth, m_Param.m_StrideX,
                            m_Param.m_PaddingMethod, m_Param.m_OutputShapeRounding);
        outHeight= CalcSize(inHeight, m_Param.m_PadTop, m_Param.m_PadBottom, m_Param.m_PoolHeight, m_Param.m_StrideY,
                            m_Param.m_PaddingMethod, m_Param.m_OutputShapeRounding);


    }
    unsigned int outChannels = inChannels;
    unsigned int outBatchSize = inBatchSize;

    TensorShape shapeOut({outBatchSize, outChannels, outHeight, outWidth});

    ConditionalThrowIfNotEqual<LayerValidationException>(
        "Pooling2dLayer: TensorShape set on OutputSlot[0] does not match the inferred shape.",
        GetOutputSlot(0).GetTensorInfo().GetShape(),
        shapeOut);
}

} // namespace armnn
