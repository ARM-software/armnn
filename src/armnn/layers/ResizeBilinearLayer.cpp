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

void ResizeBilinearLayer::ValidateTensorShapesFromInputs()
{
    ConditionalThrow<LayerValidationException>(GetInputSlot(0).GetConnection() != nullptr,
                     "MemCopyLayer: InputSlot must be connected to an OutputSlot");
    ConditionalThrow<LayerValidationException>(GetInputSlot(0).GetConnection()->IsTensorInfoSet(),
                     "MemCopyLayer: TensorInfo must be set on connected OutputSlot.");

    const TensorShape& inputShape = GetInputSlot(0).GetConnection()->GetTensorInfo().GetShape();
    unsigned int outWidth = m_Param.m_TargetWidth;
    unsigned int outHeight = m_Param.m_TargetHeight;
    unsigned int outChannels = inputShape[1];
    unsigned int outBatch = inputShape[0];
    TensorShape outShape({outBatch, outChannels, outHeight, outWidth});
    ConditionalThrowIfNotEqual<LayerValidationException>(
        "ResizeBilinearLayer: TensorShape set on OutputSlot[0] does not match the inferred shape.",
        GetOutputSlot(0).GetTensorInfo().GetShape(),
        outShape);
}

} // namespace armnn
