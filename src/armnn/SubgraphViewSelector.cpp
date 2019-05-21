//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "SubgraphViewSelector.hpp"
#include "Graph.hpp"
#include <boost/assert.hpp>
#include <algorithm>
#include <map>
#include <queue>

namespace armnn
{

namespace
{

struct LayerSelectionInfo
{
    using SplitId = uint32_t;
    using LayerInfoContainer = std::map<Layer*, LayerSelectionInfo>;
    using LayerInfoQueue = std::queue<LayerSelectionInfo*>;
    static constexpr uint32_t InitialSplitId() { return 1; }

    LayerSelectionInfo(Layer* layer, const SubgraphViewSelector::LayerSelectorFunction& selector)
    : m_Layer{layer}
    , m_SplitId{0}
    , m_IsSelected{selector(*layer)}
    , m_IsProcessed(false)
    {
        // fill topology information by storing direct children
        for (auto&& slot = m_Layer->BeginOutputSlots(); slot != m_Layer->EndOutputSlots(); ++slot)
        {
            for (InputSlot* childLayerInputSlot : slot->GetConnections())
            {
                Layer& childLayer = childLayerInputSlot->GetOwningLayer();
                m_DirectChildren.push_back(&childLayer);
            }
        }
    }

    bool IsInputLayer() const
    {
        return m_Layer->GetType() == armnn::LayerType::Input || m_Layer->GetType() == armnn::LayerType::Constant;
    }

    void CollectNonSelectedInputs(LayerSelectionInfo::LayerInfoContainer& layerInfos,
                                  SubgraphView::InputSlots& inputSlots)
    {
        for (auto&& slot = m_Layer->BeginInputSlots(); slot != m_Layer->EndInputSlots(); ++slot)
        {
            OutputSlot* parentLayerOutputSlot = slot->GetConnectedOutputSlot();
            BOOST_ASSERT_MSG(parentLayerOutputSlot != nullptr, "The input slots must be connected here.");
            if (parentLayerOutputSlot)
            {
                Layer& parentLayer = parentLayerOutputSlot->GetOwningLayer();
                auto parentInfo = layerInfos.find(&parentLayer);
                if (parentInfo == layerInfos.end() ||
                        m_SplitId != parentInfo->second.m_SplitId)
                {
                    // Avoid collecting duplicate input slots
                    InputSlot* inputSlot = &(*slot);
                    if (std::find(inputSlots.begin(), inputSlots.end(), inputSlot) == inputSlots.end())
                    {
                        inputSlots.push_back(inputSlot);
                    }
                }
            }
        }
    }

    void CollectNonSelectedOutputSlots(LayerSelectionInfo::LayerInfoContainer& layerInfos,
                                       SubgraphView::OutputSlots& outputSlots)
    {
        for (auto&& slot = m_Layer->BeginOutputSlots(); slot != m_Layer->EndOutputSlots(); ++slot)
        {
            for (InputSlot* childLayerInputSlot : slot->GetConnections())
            {
                Layer& childLayer = childLayerInputSlot->GetOwningLayer();
                auto childInfo = layerInfos.find(&childLayer);
                if (childInfo == layerInfos.end() ||
                        m_SplitId != childInfo->second.m_SplitId)
                {
                    // Avoid collecting duplicate output slots
                    OutputSlot* outputSlot = &(*slot);
                    if (std::find(outputSlots.begin(), outputSlots.end(), outputSlot) == outputSlots.end())
                    {
                        outputSlots.push_back(outputSlot);
                    }
                }
            }
        }
    }

    std::vector<Layer*> m_DirectChildren;
    Layer* m_Layer;
    SplitId m_SplitId;
    bool m_IsSelected;
    bool m_IsProcessed;
};

} // namespace <anonymous>

SubgraphViewSelector::Subgraphs
SubgraphViewSelector::SelectSubgraphs(Graph& graph, const LayerSelectorFunction& selector)
{
    SubgraphView subgraph(graph);
    return SubgraphViewSelector::SelectSubgraphs(subgraph, selector);
}


template<typename Delegate>
void ForEachLayerInput(LayerSelectionInfo::LayerInfoContainer& layerInfos,
                       LayerSelectionInfo& layerInfo,
                       Delegate function)
{
    Layer& layer = *layerInfo.m_Layer;

    for (auto inputSlot : layer.GetInputSlots())
    {
        auto connectedInput = boost::polymorphic_downcast<OutputSlot*>(inputSlot.GetConnection());
        BOOST_ASSERT_MSG(connectedInput, "Dangling input slot detected.");
        Layer& inputLayer = connectedInput->GetOwningLayer();

        auto parentInfo = layerInfos.find(&inputLayer);
        if (parentInfo != layerInfos.end())
        {
            function(parentInfo->second);
        }
    }
}

template<typename Delegate>
void ForEachLayerOutput(LayerSelectionInfo::LayerInfoContainer& layerInfos,
                        LayerSelectionInfo& layerInfo,
                        Delegate function)
{
    Layer& layer= *layerInfo.m_Layer;

    for (auto& outputSlot : layer.GetOutputSlots())
    {
        for (auto& output : outputSlot.GetConnections())
        {
            Layer& childLayer = output->GetOwningLayer();

            auto childInfo = layerInfos.find(&childLayer);
            if (childInfo != layerInfos.end())
            {
                function(childInfo->second);
            }
        }
    }
}

void AssignSplitId(LayerSelectionInfo::LayerInfoContainer& layerInfos, LayerSelectionInfo& layerInfo)
{
    bool newSplit = false;
    LayerSelectionInfo::SplitId minSplitId = std::numeric_limits<LayerSelectionInfo::SplitId>::max();
    LayerSelectionInfo::SplitId maxSplitId = std::numeric_limits<LayerSelectionInfo::SplitId>::lowest();
    LayerSelectionInfo::SplitId maxSelectableId = std::numeric_limits<LayerSelectionInfo::SplitId>::lowest();

    ForEachLayerInput(layerInfos, layerInfo, [&newSplit, &minSplitId, &maxSplitId, &maxSelectableId, &layerInfo](
        LayerSelectionInfo& parentInfo)
        {
            minSplitId = std::min(minSplitId, parentInfo.m_SplitId);
            maxSplitId = std::max(maxSplitId, parentInfo.m_SplitId);
            if (parentInfo.m_IsSelected && layerInfo.m_IsSelected)
            {
                maxSelectableId = std::max(maxSelectableId, parentInfo.m_SplitId);
            }

            if (layerInfo.m_IsSelected != parentInfo.m_IsSelected)
            {
                newSplit = true;
            }

        });

    // Assign the split Id for the current layerInfo
    if (newSplit)
    {
        if (maxSelectableId > minSplitId)
        {
            // We can be overly aggressive when choosing to create a new split so
            // here we determine if one of the parent branches are suitable candidates for continuation instead.
            // Any splitId > minSplitId will come from a shorter branch...and therefore should not be from
            // the split containing the original fork and thus we avoid the execution dependency.
            layerInfo.m_SplitId = maxSelectableId;
        }
        else
        {
            layerInfo.m_SplitId = ++maxSplitId;
        }
    } else
    {
        // The branch with the highest splitId represents the shortest path of selected nodes.
        layerInfo.m_SplitId = maxSplitId;
    }
}

bool IsReadyForSplitAssignment(LayerSelectionInfo::LayerInfoContainer& layerInfos, LayerSelectionInfo& layerInfo)
{
    bool ready = true;
    ForEachLayerInput(layerInfos, layerInfo,
                      [&ready](LayerSelectionInfo& parentInfo)
                          {
                              if (!parentInfo.m_IsProcessed)
                              {
                                  ready = false;
                              }
                          });
    return ready;
}

SubgraphViewSelector::Subgraphs
SubgraphViewSelector::SelectSubgraphs(SubgraphView& subgraph, const LayerSelectorFunction& selector)
{
    LayerSelectionInfo::LayerInfoContainer layerInfos;

    LayerSelectionInfo::LayerInfoQueue processQueue;
    for (auto& layer : subgraph)
    {
        auto emplaced = layerInfos.emplace(layer, LayerSelectionInfo{layer, selector});
        LayerSelectionInfo& layerInfo = emplaced.first->second;

        // Start with Input type layers
        if (layerInfo.IsInputLayer())
        {
            processQueue.push(&layerInfo);
        }
    }

    const SubgraphView::InputSlots& subgraphInputSlots = subgraph.GetInputSlots();
    for (auto& inputSlot : subgraphInputSlots)
    {
        Layer& layer = inputSlot->GetOwningLayer();
        auto emplaced = layerInfos.emplace(&layer, LayerSelectionInfo{&layer, selector});
        LayerSelectionInfo& layerInfo = emplaced.first->second;

        processQueue.push(&layerInfo);
    }

    while (!processQueue.empty())
    {
        LayerSelectionInfo& layerInfo = *processQueue.front();
        processQueue.pop(); // remove front from queue

        // This layerInfo may have been added to the queue multiple times, so skip if we have already processed it
        if (!layerInfo.m_IsProcessed)
        {

            // Only process this layerInfo if all inputs have been processed
            if (!IsReadyForSplitAssignment(layerInfos, layerInfo))
            {
                // Put back of the process queue if we can't process it just yet
                processQueue.push(&layerInfo);
                continue; // Skip to next iteration
            }

            // Now we do the processing
            AssignSplitId(layerInfos, layerInfo);

            // Queue any child nodes for processing
            ForEachLayerOutput(layerInfos, layerInfo, [&processQueue](LayerSelectionInfo& childInfo)
                {
                    processQueue.push(&childInfo);
                });

            // We don't need to process this node again
            layerInfo.m_IsProcessed = true;
        }
    }

    // Collect all selected layers keyed by split id into a map
    using SelectionInfoPtrs = std::vector<LayerSelectionInfo*>;
    std::map<uint32_t, SelectionInfoPtrs> splitMap;
    for (auto& info : layerInfos)
    {
        if (info.second.m_IsSelected)
        {
            auto it = splitMap.find(info.second.m_SplitId);
            if (it == splitMap.end())
            {
                splitMap.insert(std::make_pair(info.second.m_SplitId, SelectionInfoPtrs{&info.second}));
            }
            else
            {
                it->second.push_back(&info.second);
            }
        }
    }

    // Now each non-empty split id represents a subgraph
    Subgraphs result;
    for (auto& splitGraph : splitMap)
    {
        if (splitGraph.second.empty() == false)
        {
            SubgraphView::InputSlots inputs;
            SubgraphView::OutputSlots outputs;
            SubgraphView::Layers layers;
            for (auto&& infoPtr : splitGraph.second)
            {
                infoPtr->CollectNonSelectedInputs(layerInfos, inputs);
                infoPtr->CollectNonSelectedOutputSlots(layerInfos, outputs);
                layers.push_back(infoPtr->m_Layer);
            }
            // Create a new sub-graph with the new lists of input/output slots and layer
            result.emplace_back(std::make_unique<SubgraphView>(std::move(inputs),
                                                               std::move(outputs),
                                                               std::move(layers)));
        }
    }

    return result;
}

} // namespace armnn
