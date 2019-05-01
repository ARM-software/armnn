//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "SubgraphViewSelector.hpp"
#include "Graph.hpp"
#include <boost/assert.hpp>
#include <algorithm>
#include <unordered_map>

namespace armnn
{

namespace
{

struct LayerSelectionInfo
{
    using LayerInfoContainer = std::unordered_map<Layer*, LayerSelectionInfo>;
    static constexpr uint32_t InitialSplitId() { return 1; }

    LayerSelectionInfo(Layer* layer, const SubgraphViewSelector::LayerSelectorFunction& selector)
    : m_Layer{layer}
    , m_SplitId{0}
    , m_IsSelected{selector(*layer)}
    , m_IsVisited(false)
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

    void MarkChildrenSplits(LayerInfoContainer& network,
                            uint32_t splitId,
                            bool prevSelected)
    {
        if (m_IsVisited)
        {
            return;
        }
        m_IsVisited = true;

        if (m_SplitId < splitId)
        {
            m_SplitId = splitId;
        }

        // introduce a new split point at all non-selected points, but only if the
        // previous point was selected. this prevents creating a new subgraph at
        // every non-selected layer
        if (!m_IsSelected && prevSelected)
        {
            ++m_SplitId;
        }

        for (auto& layer : m_DirectChildren)
        {
            auto it = network.find(layer);
            BOOST_ASSERT_MSG(it != network.end(), "All layers must be part of the topology.");
            if (it != network.end())
            {
                it->second.MarkChildrenSplits(network, m_SplitId, m_IsSelected);
            }
        }
    }

    bool IsInputLayer() const
    {
        return m_Layer->GetType() == armnn::LayerType::Input;
    }

    void CollectNonSelectedInputs(SubgraphView::InputSlots& inputSlots,
                                  const SubgraphViewSelector::LayerSelectorFunction& selector)
    {
        for (auto&& slot = m_Layer->BeginInputSlots(); slot != m_Layer->EndInputSlots(); ++slot)
        {
            OutputSlot* parentLayerOutputSlot = slot->GetConnectedOutputSlot();
            BOOST_ASSERT_MSG(parentLayerOutputSlot != nullptr, "The input slots must be connected here.");
            if (parentLayerOutputSlot)
            {
                Layer& parentLayer = parentLayerOutputSlot->GetOwningLayer();
                if (selector(parentLayer) == false)
                {
                    inputSlots.push_back(&(*slot));
                }
            }
        }
    }

    void CollectNonSelectedOutputSlots(SubgraphView::OutputSlots& outputSlots,
                                       const SubgraphViewSelector::LayerSelectorFunction& selector)
    {
        for (auto&& slot = m_Layer->BeginOutputSlots(); slot != m_Layer->EndOutputSlots(); ++slot)
        {
            for (InputSlot* childLayerInputSlot : slot->GetConnections())
            {
                Layer& childLayer = childLayerInputSlot->GetOwningLayer();
                if (selector(childLayer) == false)
                {
                    outputSlots.push_back(&(*slot));
                }
            }
        }
    }

    std::vector<Layer*> m_DirectChildren;
    Layer* m_Layer;
    uint32_t m_SplitId;
    bool m_IsSelected;
    bool m_IsVisited;
};

} // namespace <anonymous>

SubgraphViewSelector::Subgraphs
SubgraphViewSelector::SelectSubgraphs(Graph& graph, const LayerSelectorFunction& selector)
{
    SubgraphView subgraph(graph);
    return SubgraphViewSelector::SelectSubgraphs(subgraph, selector);
}

SubgraphViewSelector::Subgraphs
SubgraphViewSelector::SelectSubgraphs(SubgraphView& subgraph, const LayerSelectorFunction& selector)
{
    LayerSelectionInfo::LayerInfoContainer layerInfo;

    for (auto& layer : subgraph)
    {
        layerInfo.emplace(layer, LayerSelectionInfo{layer, selector});
    }

    uint32_t splitNo = LayerSelectionInfo::InitialSplitId();
    for (auto& info : layerInfo)
    {
        if (info.second.IsInputLayer())
        {
            // For each input layer we mark the graph where subgraph
            // splits need to happen because of the dependency between
            // the selected and non-selected nodes
            info.second.MarkChildrenSplits(layerInfo, splitNo, false);
        }
    }

    // Collect all selected layers keyed by split id into a map
    using SelectionInfoPtrs = std::vector<LayerSelectionInfo*>;
    std::unordered_map<uint32_t, SelectionInfoPtrs> splitMap;
    for (auto& info : layerInfo)
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
                infoPtr->CollectNonSelectedInputs(inputs, selector);
                infoPtr->CollectNonSelectedOutputSlots(outputs, selector);
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
