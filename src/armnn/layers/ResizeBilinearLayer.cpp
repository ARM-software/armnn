//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//
#include "ResizeBilinearLayer.hpp"

#include "LayerCloneBase.hpp"

#include <armnn/TypesUtils.hpp>
#include <backends/WorkloadData.hpp>
#include <backends/WorkloadFactory.hpp>

namespace armnn
{

ResizeBilinearLayer::ResizeBilinearLayer(const ResizeBilinearDescriptor& param, const char* name)
    : LayerWithParameters(1, 1, LayerType::ResizeBilinear, param, name)
{
}

std::unique_ptr<IWorkload> ResizeBilinearLayer::CreateWorkload(const Graph& graph,
                                                               const IWorkloadFactory& factory) const
{
    ResizeBilinearQueueDescriptor descriptor;
    return factory.CreateResizeBilinear(descriptor, PrepInfoAndDesc(descriptor, graph));
}

ResizeBilinearLayer* ResizeBilinearLayer::Clone(Graph& graph) const
{
    return CloneBase<ResizeBilinearLayer>(graph, m_Param, GetName());
}

std::vector<TensorShape> ResizeBilinearLayer::InferOutputShapes(const std::vector<TensorShape>& inputShapes) const
{
    BOOST_ASSERT(inputShapes.size() == 1);
    const TensorShape& inputShape = inputShapes[0];

    unsigned int outWidth = m_Param.m_TargetWidth;
    unsigned int outHeight = m_Param.m_TargetHeight;
    unsigned int outChannels = inputShape[1];
    unsigned int outBatch = inputShape[0];

    return std::vector<TensorShape>({ TensorShape({outBatch, outChannels, outHeight, outWidth}) });
}

void ResizeBilinearLayer::ValidateTensorShapesFromInputs()
{
    VerifyLayerConnections(1, CHECK_LOCATION());

    auto inferredShapes = InferOutputShapes({ GetInputSlot(0).GetConnection()->GetTensorInfo().GetShape() });

    BOOST_ASSERT(inferredShapes.size() == 1);

    ConditionalThrowIfNotEqual<LayerValidationException>(
        "ResizeBilinearLayer: TensorShape set on OutputSlot[0] does not match the inferred shape.",
        GetOutputSlot(0).GetTensorInfo().GetShape(),
        inferredShapes[0]);
}

} // namespace armnn
