//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "PermuteAndBatchToSpaceAsDepthToSpace.hpp"

using namespace armnn;
using namespace armnn::optimizations;

void PermuteAndBatchToSpaceAsDepthToSpaceImpl::Run(Graph& graph, InputSlot& connection) const
{
    // Validate base layer (the Permute) is compatible
    Layer& base = connection.GetConnectedOutputSlot()->GetOwningLayer();
    BOOST_ASSERT(base.GetType() == LayerType::Permute);
    const TensorInfo& inputInfo        = base.GetInputSlot(0).GetConnection()->GetTensorInfo();
    const TensorInfo& intermediateInfo = base.GetOutputSlot(0).GetTensorInfo();
    if (intermediateInfo.GetNumDimensions() != 4)
    {
        // Must be 4D, otherwise the below checks do not make sense
        return;
    }
    if (!static_cast<PermuteLayer&>(base).GetParameters().m_DimMappings.IsEqual(PermutationVector{ 3, 1, 2, 0 }))
    {
        // Must swap batch and channels dimensions, otherwise it is not the (original) channels dimension
        // that is being decomposed.
        return;
    }

    // Validate child layer (the BatchToSpace) is compatible
    Layer& child = connection.GetOwningLayer();
    BOOST_ASSERT(child.GetType() == LayerType::BatchToSpaceNd);
    const TensorInfo& outputInfo                     = child.GetOutputSlot(0).GetTensorInfo();
    const BatchToSpaceNdDescriptor& batchToSpaceDesc = static_cast<BatchToSpaceNdLayer&>(child).GetParameters();
    if (batchToSpaceDesc.m_DataLayout != DataLayout::NHWC)
    {
        // The rest of this function assumes NHWC, although in future this restriction could be lifted.
        return;
    }
    if (batchToSpaceDesc.m_Crops != std::vector<std::pair<unsigned int, unsigned int>>{ { 0, 0 }, { 0, 0 } })
    {
        // Cropping is not supported in DepthToSpace
        return;
    }
    if (batchToSpaceDesc.m_BlockShape.size() != 2 ||
        batchToSpaceDesc.m_BlockShape[0] != batchToSpaceDesc.m_BlockShape[1])
    {
        // Asymmetric or non-2D block sizes are not supported by DepthToSpace
        return;
    }
    uint32_t blockSize = batchToSpaceDesc.m_BlockShape[0];
    if (outputInfo.GetShape()[0] != 1 || outputInfo.GetShape()[3] != 1)
    {
        // The final output must have 1 batch and 1 channel because these dimensions will be swapped around
        // once we make the substitution, and it needs to be equivalent.
        return;
    }

    // Validate the intermediate tensor quantization params.
    // These must be identical to either the input or output quantization params, otherwise the intermediate tensor
    // may not have sufficient range/precision to preserve the values.
    // This would mean that once we perform the substitution this loss of precision will no longer occur,
    // so we would have changed the meaning of the network.
    bool isIntermediateQuantParamsSameAsInput =
        intermediateInfo.GetQuantizationScale() == inputInfo.GetQuantizationScale() &&
        intermediateInfo.GetQuantizationOffset() == inputInfo.GetQuantizationOffset();
    bool isIntermediateQuantParamsSameAsOutput =
        intermediateInfo.GetQuantizationScale() == outputInfo.GetQuantizationScale() &&
        intermediateInfo.GetQuantizationOffset() == outputInfo.GetQuantizationOffset();
    if (!isIntermediateQuantParamsSameAsInput && !isIntermediateQuantParamsSameAsOutput)
    {
        return;
    }

    // Insert equivalent DepthToSpace layer
    const std::string name = std::string("merged-") + base.GetName() + std::string("-with-") + child.GetName();

    // Inserts equivalent reshape before base layer.
    const DepthToSpaceDescriptor depthToSpaceDesc(blockSize, DataLayout::NHWC);
    auto& depthToSpace = *graph.InsertNewLayer<DepthToSpaceLayer>(base.GetInputSlot(0), depthToSpaceDesc, name.c_str());
    depthToSpace.GetOutputHandler().SetTensorInfo(outputInfo);

    // Moves connections from child output to new layer.
    // Child layer will be removed as it's left unconnected.
    // Base layer will be removed if left unconnected.
    child.GetOutputSlot().MoveAllConnections(depthToSpace.GetOutputSlot());
}
