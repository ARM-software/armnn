//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include "MergerLayer.hpp"
#include "LayerCloneBase.hpp"

#include <armnn/TypesUtils.hpp>
#include <backendsCommon/WorkloadData.hpp>
#include <backendsCommon/WorkloadFactory.hpp>

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

    // Copies the view origins to the descriptor.
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
    //If sub tensors are supported then the merger
    //just needs to make sure that the outputs of the prev layer
    //are made subtensors of the output of the merger layer.
    m_OutputHandlers[0].CreateTensorHandles(factory);

    if (factory.SupportsSubTensors())
    {
        std::queue<MergerLayer*> m_MergerLayers;

        m_MergerLayers.push(this);
        while (!m_MergerLayers.empty())
        {
            MergerLayer* currentLayer = m_MergerLayers.front();
            ITensorHandle* parentTensor = currentLayer->GetOutputHandler(0).GetData();
            const TensorInfo& parentInfo = currentLayer->GetOutputHandler(0).GetTensorInfo();
            m_MergerLayers.pop();

            const unsigned int numInputSlots = currentLayer->GetNumInputSlots();

            // First go through all the input slots and verify that we can sub-tensor all the inputs.
            std::vector<std::unique_ptr<ITensorHandle>> subTensors(0);
            subTensors.reserve(numInputSlots);
            for (unsigned int i = 0; i < numInputSlots; ++i)
            {
                OutputSlot* slot = currentLayer->GetInputSlot(i).GetConnectedOutputSlot();
                const TensorInfo& info = slot->GetTensorInfo();

                auto CreateSubTensor = [&]()
                {
                    // Make sure quantization parameters are in the same space
                    if (parentInfo.IsTypeSpaceMatch(info))
                    {
                        return factory.CreateSubTensorHandle(*parentTensor,
                                                             info.GetShape(),
                                                             currentLayer->m_Param.GetViewOrigin(i));
                    }
                    return std::unique_ptr<ITensorHandle>();
                };

                auto subTensor = CreateSubTensor();
                if (!subTensor)
                {
                    break; //Failed to create a valid sub-tensor, so stop trying with the rest of the inputs.
                }
                else
                {
                    subTensors.push_back(std::move(subTensor)); // store the valid sub-tensor.
                }
            }

            // Ensure that ALL inputs can be substituted with valid sub-tensors
            if (subTensors.size() < numInputSlots)
            {
                continue; // Don't optimize this Merge layer with sub-tensors
            }

            // Substitute input tensors with sub-tensors by replacing the output tensors on the connected layers.
            unsigned int i=0;
            for (auto& subTensor : subTensors)
            {
                OutputSlot* slot = currentLayer->GetInputSlot(i).GetConnectedOutputSlot();
                OutputHandler& outputHandler = slot->GetOutputHandler();

                BOOST_ASSERT_MSG(subTensor, "MergerLayer: Expected a valid sub-tensor for substitution.");
                outputHandler.SetData(std::move(subTensor));

                Layer& inputLayer = slot->GetOwningLayer();
                if (inputLayer.GetType() == LayerType::Merger)
                {
                    // Continue with the substitution if the connected inputs are also merger layers
                    m_MergerLayers.push(boost::polymorphic_downcast<MergerLayer*>(&inputLayer));
                }
                ++i;
            }
        }
    }
}

MergerLayer* MergerLayer::Clone(Graph& graph) const
{
    return CloneBase<MergerLayer>(graph, m_Param, GetName());
}

std::vector<TensorShape> MergerLayer::InferOutputShapes(const std::vector<TensorShape>& inputShapes) const
{
    BOOST_ASSERT(inputShapes.size() == m_Param.GetNumViews());

    unsigned int numDims = m_Param.GetNumDimensions();
    for (unsigned int i=0; i< inputShapes.size(); i++)
    {
        auto& inputShape = inputShapes[i];

        ConditionalThrowIfNotEqual<LayerValidationException>(
            "MergerLayer: Num Dimensions must match all inputs.",
            numDims,
            inputShape.GetNumDimensions());
    }

    // Finds the bounding box (extents) of all the views.
    std::vector<unsigned int> extentMin(numDims);
    std::vector<unsigned int> extentMax(numDims);
    for (unsigned int i = 0; i < inputShapes.size(); i++)
    {
        const uint32_t* origin = m_Param.GetViewOrigin(i);
        const armnn::TensorShape& shape = inputShapes[i];
        for (unsigned int d = 0; d < numDims; d++)
        {
            extentMin[d] = std::min(extentMin[d], origin[d]);
            extentMax[d] = std::max(extentMax[d], origin[d] + shape[d]);
        }
    }

    // Checks that the bounding box starts at the origin.
    if (!std::all_of(extentMin.begin(), extentMin.end(), [](unsigned int s) { return s == 0; }))
    {
        throw LayerValidationException("MergerLayer: there is no view that starts at the origin");
    }

    // Checks that there are no overlaps of views (this would lead to undefined output at those locations).
    // Checks each pair of views against each other
    // (and doesn't bother to check against self, or check the same pair both ways round).
    for (unsigned int a = 0; a < inputShapes.size(); a++)
    {
        const uint32_t* aOrigin = m_Param.GetViewOrigin(a);
        const armnn::TensorShape& aShape = inputShapes[a];
        for (unsigned int b = 0; b < a; b++)
        {
            const uint32_t* bOrigin = m_Param.GetViewOrigin(b);
            const armnn::TensorShape& bShape = inputShapes[b];

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

    // Checks that there are no "holes", i.e. regions of the output which is not covered by a view.
    // Because we already checked that there are no overlaps, this can be done simply by checking that
    // the total 'volume' of the views is the same as the output.
    unsigned int totalViewsVolume = 0;
    for (unsigned int i = 0; i < inputShapes.size(); i++)
    {
        totalViewsVolume += inputShapes[i].GetNumElements();
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

    return std::vector<TensorShape>({ TensorShape({numDims, extentMax.data()}) });
}

void MergerLayer::ValidateTensorShapesFromInputs()
{
    // Validates Merger layer.
    ConditionalThrowIfNotEqual<LayerValidationException>(
        "MergerLayer: Num Inputs must match num views.",
        m_Param.GetNumViews(),
        GetNumInputSlots());

    VerifyLayerConnections(m_Param.GetNumViews(), CHECK_LOCATION());

    std::vector<TensorShape> inputShapes;
    for (unsigned int i = 0; i < GetNumInputSlots(); ++i)
    {
        inputShapes.push_back(GetInputSlot(i).GetConnection()->GetTensorInfo().GetShape());
    }

    auto inferredShapes = InferOutputShapes(inputShapes);

    BOOST_ASSERT(inferredShapes.size() == 1);

    ConditionalThrowIfNotEqual<LayerValidationException>(
        "MergerLayer: TensorShape set on OutputSlot[0] does not match the inferred shape.",
        GetOutputSlot(0).GetTensorInfo().GetShape(),
        inferredShapes[0]);
}

void MergerLayer::Accept(ILayerVisitor& visitor) const
{
    visitor.VisitMergerLayer(this, GetParameters(), GetName());
}

} // namespace armnn armnn
