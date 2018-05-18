//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//
#include "MergerLayer.hpp"
#include "LayerCloneBase.hpp"

#include <armnn/TypesUtils.hpp>
#include <backends/WorkloadData.hpp>
#include <backends/WorkloadFactory.hpp>

#include <queue>

namespace armnn
{

MergerLayer::MergerLayer(const OriginsDescriptor& param, const char* name)
    : LayerWithParameters(param.GetNumViews(), 1, LayerType::Merger, param, name)
{
}

std::unique_ptr<IWorkload> MergerLayer::CreateWorkload(const Graph& graph, const IWorkloadFactory& factory) const
{
    MergerQueueDescriptor descriptor;

    // copy the view origins to the descriptor
    descriptor.m_ViewOrigins.reserve(m_Param.GetNumViews());
    for (unsigned int i = 0; i < m_Param.GetNumViews(); ++i)
    {
        descriptor.m_ViewOrigins.emplace_back(
            std::vector<unsigned int>(m_Param.GetViewOrigin(i), m_Param.GetViewOrigin(i) + m_Param.GetNumDimensions()));
    }

    return factory.CreateMerger(descriptor, PrepInfoAndDesc(descriptor, graph));
}

void MergerLayer::CreateTensorHandles(Graph& graph, const IWorkloadFactory& factory)
{
    //if sub tensors are supported than the merger
    //just needs to make sure that the outputs of the prev layer
    //are made subtensors of the output of the merger layer
    m_OutputHandlers[0].CreateTensorHandles(factory);
    if (factory.SupportsSubTensors())
    {
        std::queue<MergerLayer*> m_MergerLayers;

        m_MergerLayers.push(this);
        while (!m_MergerLayers.empty())
        {
            MergerLayer* currentLayer = m_MergerLayers.front();
            ITensorHandle* parentTensor = currentLayer->GetOutputHandler(0).GetData();

            m_MergerLayers.pop();

            const unsigned int numInputSlots = currentLayer->GetNumInputSlots();
            for (unsigned int i = 0; i < numInputSlots; ++i)
            {
                OutputSlot* slot = currentLayer->GetInputSlot(i).GetConnectedOutputSlot();
                OutputHandler& outputHandler = slot->GetOutputHandler();
                outputHandler.SetData(factory.CreateSubTensorHandle(*parentTensor,
                                                                    outputHandler.GetTensorInfo().GetShape(),
                                                                    currentLayer->m_Param.GetViewOrigin(i)));

                Layer& inputLayer = slot->GetOwningLayer();
                if (inputLayer.GetType() == LayerType::Merger)
                {
                    m_MergerLayers.push(boost::polymorphic_downcast<MergerLayer*>(&inputLayer));
                }
            }
        }
    }
}

MergerLayer* MergerLayer::Clone(Graph& graph) const
{
    return CloneBase<MergerLayer>(graph, m_Param, GetName());
}

void MergerLayer::ValidateTensorShapesFromInputs()
{
    // Validate Merger layer
    ConditionalThrowIfNotEqual<LayerValidationException>(
        "MergerLayer: Num Inputs must match num views.",
        m_Param.GetNumViews(),
        GetNumInputSlots());

    unsigned int numDims = m_Param.GetNumDimensions();
    for (unsigned int i=0; i<GetNumInputSlots(); i++)
    {
        auto& inputInfo = GetInputSlot(i).GetConnection()->GetTensorInfo();

        boost::ignore_unused(inputInfo);
        ConditionalThrowIfNotEqual<LayerValidationException>(
            "MergerLayer: Num Dimensions must match all inputs.",
            numDims,
            inputInfo.GetNumDimensions());
    }

    // Find the bounding box (extents) of all the views
    std::vector<unsigned int> extentMin(numDims);
    std::vector<unsigned int> extentMax(numDims);
    for (unsigned int i = 0; i < GetNumInputSlots(); i++)
    {
        const uint32_t* origin = m_Param.GetViewOrigin(i);
        const armnn::TensorShape& shape = GetInputSlot(i).GetConnection()->GetTensorInfo().GetShape();
        for (unsigned int d = 0; d < numDims; d++)
        {
            extentMin[d] = std::min(extentMin[d], origin[d]);
            extentMax[d] = std::max(extentMax[d], origin[d] + shape[d]);
        }
    }

    // Check that the bounding box starts at the origin
    if (!std::all_of(extentMin.begin(), extentMin.end(), [](unsigned int s) { return s == 0; }))
    {
        throw LayerValidationException("MergerLayer: there is no view that starts at the origin");
    }

    // Check that there are no overlaps of views (this would lead to undefined output at those locations).
    // Check each pair of views against each other
    // (and don't bother to check against self, or check the same pair both ways round)
    for (unsigned int a = 0; a < GetNumInputSlots(); a++)
    {
        const uint32_t* aOrigin = m_Param.GetViewOrigin(a);
        const armnn::TensorShape& aShape = GetInputSlot(a).GetConnection()->GetTensorInfo().GetShape();
        for (unsigned int b = 0; b < a; b++)
        {
            const uint32_t* bOrigin = m_Param.GetViewOrigin(b);
            const armnn::TensorShape& bShape = GetInputSlot(b).GetConnection()->GetTensorInfo().GetShape();

            bool allAxesOverlap = true;
            for (unsigned int d = 0; d < numDims && allAxesOverlap; d++)
            {
                unsigned int a1 = aOrigin[d];
                unsigned int a2 = aOrigin[d] + aShape[d];

                unsigned int b1 = bOrigin[d];
                unsigned int b2 = bOrigin[d] + bShape[d];

                if (a2 <= b1 || b2 <= a1)
                {
                    allAxesOverlap = false;
                }
            }
            if (allAxesOverlap)
            {
                throw LayerValidationException("MergerLayer: Some views overlap.");
            }
        }
    }

    // Check that there are no "holes", i.e. regions of the output which is not covered by a view.
    // Because we already checked that there are no overlaps, this can be done simply by checking that
    // the total 'volume' of the views is the same as the output.
    unsigned int totalViewsVolume = 0;
    for (unsigned int i = 0; i < GetNumInputSlots(); i++)
    {
        totalViewsVolume += GetInputSlot(i).GetConnection()->GetTensorInfo().GetNumElements();
    }
    unsigned int outputVolume = 1;
    for (unsigned int d = 0; d < numDims; d++)
    {
        outputVolume *= (extentMax[d] - extentMin[d]);
    }

    ConditionalThrowIfNotEqual<LayerValidationException>(
        "MergerLayer: there are some gaps between views",
        totalViewsVolume,
        outputVolume);

    TensorShape outShape(numDims, extentMax.data());
    ConditionalThrowIfNotEqual<LayerValidationException>(
        "MergerLayer: TensorShape set on OutputSlot[0] does not match the inferred shape.",
        GetOutputSlot(0).GetTensorInfo().GetShape(),
        outShape);
}

} // namespace armnn armnn
