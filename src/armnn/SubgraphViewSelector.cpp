//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "SubgraphViewSelector.hpp"
#include "Graph.hpp"

#include <armnn/utility/Assert.hpp>
#include <armnn/utility/IgnoreUnused.hpp>
#include <armnn/utility/PolymorphicDowncast.hpp>

#include <algorithm>
#include <map>
#include <queue>
#include <unordered_set>

namespace armnn
{

namespace
{

/// Intermediate data-structure to store the subgraph that a layer has been assigned to.
/// This is a "disjoint set" data structure that allows efficient merging of subgraphs,
/// which is a key part of the algorithm. Subgraphs are arranged in singly-linked trees
/// (with each node storing a pointer to its parent). Subgraphs in the same tree are considered
/// to have been merged. Merging subgraphs is performed by attaching one tree to another,
/// which is a simple pointer update.
///
/// NOTE: Due to the way this is stored, it is almost never correct to directly compare pointers
/// to two PartialSubgraphs to check if two layers belong in the same subgraph. Instead you
/// should use IsMergedWith().
///
/// This structure also stores information about the dependencies of each subgraph, which is needed
/// to determine whether certain subgraphs can be merged. Checking whether a subgraph
/// depends on another subgraph is a frequent operation in the algorithm (see AssignSplitId) and so this is optimized
/// in preference to the merging of subgraphs. This leads to an approach where each subgraph stores
/// a set of all the subgraphs it depends on (for a fast lookup). In order to efficiently update this
/// set as subgraphs are merged means we also store a set of subgraphs which *depend on us* (i.e. the
/// complement of our dependencies).
class PartialSubgraph
{
public:
    /// If this subgraph has been merged with another then there is an agreed "representative" for the combined
    /// subgraph, which uniquely identifies the subgraph.
    PartialSubgraph* GetRepresentative()
    {
        // Recurse up the tree to find the root node.
        if (m_Parent == nullptr)
        {
            return this;
        }
        else
        {
            PartialSubgraph* result = m_Parent->GetRepresentative();
            // Update our parent pointer to point directly to the root in order to speed up future calls to this method.
            // This essentially "flattens" the tree.
            m_Parent = result;
            return result;
        }
    }

    /// Merges this subgraph with another.
    void MergeWith(PartialSubgraph* other)
    {
        if (m_Parent == nullptr)
        {
            other = other->GetRepresentative();
            if (this == other)
            {
                // Already merged - no-op
                return;
            }
            m_Parent = other;

            // Update others' dependency sets to point to the new representative rather than us.
            // Keeping these up-to-date means we can rely on these sets containing representatives when
            // we perform a lookup in HasAntecedent() and so don't need to resolve the representative for each element
            // of the set. See description at the top of this class for more rationale.
            for (PartialSubgraph* a : m_Antecedents)
            {
                size_t numErased = a->m_Dependants.erase(this);
                ARMNN_ASSERT(numErased == 1);
                IgnoreUnused(numErased);
                a->m_Dependants.insert(m_Parent);
            }
            for (PartialSubgraph* a : m_Dependants)
            {
                size_t numErased = a->m_Antecedents.erase(this);
                ARMNN_ASSERT(numErased == 1);
                IgnoreUnused(numErased);
                a->m_Antecedents.insert(m_Parent);
            }

            // Merge our dependency sets into our new representative.
            // We no longer need to maintain our own sets, as requests will always be forwarded to the representative.
            m_Parent->m_Antecedents.insert(m_Antecedents.begin(), m_Antecedents.end());
            m_Antecedents.clear();
            m_Parent->m_Dependants.insert(m_Dependants.begin(), m_Dependants.end());
            m_Dependants.clear();
        }
        else
        {
            // Defer request to the representative
            GetRepresentative()->MergeWith(other);
        }
    }

    /// Checks if this subgraph has been merged with the given subgraph.
    bool IsMergedWith(PartialSubgraph* other)
    {
        return GetRepresentative() == other->GetRepresentative();
    }

    /// Marks the given subgraph as a direct antecedent (dependency) of this one.
    void AddDirectAntecedent(PartialSubgraph* antecedent)
    {
        if (m_Parent == nullptr)
        {
            antecedent = antecedent->GetRepresentative();

            m_Antecedents.insert(antecedent);
            // Also record all of its antecedents, so that we end up with direct and indirect antecedents.
            // This makes the lookup in HasAntecedent() faster.
            m_Antecedents.insert(antecedent->m_Antecedents.begin(), antecedent->m_Antecedents.end());
            // All of our dependents also need to include the new antecedents
            for (PartialSubgraph* d : m_Dependants)
            {
                d->m_Antecedents.insert(antecedent);
                d->m_Antecedents.insert(antecedent->m_Antecedents.begin(), antecedent->m_Antecedents.end());
            }

            // Store reverse dependencies as well, required so that we can efficiently navigate the graph
            // when making updates.
            antecedent->m_Dependants.insert(this);
            antecedent->m_Dependants.insert(m_Dependants.begin(), m_Dependants.end());
            for (PartialSubgraph* a : antecedent->m_Antecedents)
            {
                a->m_Dependants.insert(this);
                a->m_Dependants.insert(m_Dependants.begin(), m_Dependants.end());
            }
        }
        else
        {
            // Defer request to the representative
            GetRepresentative()->AddDirectAntecedent(antecedent);
        }
    }

    /// Checks if this subgraph is dependent on the given subgraph, either directly or indirectly.
    bool HasAntecedent(PartialSubgraph* antecedent)
    {
        if (m_Parent == nullptr)
        {
            antecedent = antecedent->GetRepresentative();
            // Thanks to keeping this set updated in MergeWith and AddDirectAntecedent, we can do an efficient lookup.
            return m_Antecedents.count(antecedent) > 0;
        }
        else
        {
            // Defer request to the representative
            return GetRepresentative()->HasAntecedent(antecedent);
        }
    }

private:
    /// Pointer to the parent node in the tree. If this is null then we are the representative for our merged subgraph.
    PartialSubgraph* m_Parent;
    /// The representatives of all the subgraphs which we depend on, either directly or indirectly.
    std::unordered_set<PartialSubgraph*> m_Antecedents;
    /// The representatives of all the subgraphs which depend on us, either directly or indirectly.
    std::unordered_set<PartialSubgraph*> m_Dependants;
};

/// Intermediate data structure to store information associated with a particular layer.
struct LayerSelectionInfo
{
    using LayerInfoContainer = std::map<IConnectableLayer*, LayerSelectionInfo>;
    using LayerInfoQueue = std::queue<LayerSelectionInfo*>;

    LayerSelectionInfo(Layer* layer, const SubgraphViewSelector::LayerSelectorFunction& selector)
    : m_Layer{layer}
    , m_Subgraph{nullptr}
    , m_IsSelected{selector(*layer)}
    , m_IsProcessed(false)
    {
    }

    bool IsInputLayer() const
    {
        return m_Layer->GetType() == armnn::LayerType::Input || m_Layer->GetType() == armnn::LayerType::Constant;
    }

    void CollectNonSelectedInputs(LayerSelectionInfo::LayerInfoContainer& layerInfos,
                                  SubgraphView::IInputSlots& inputSlots)
    {
        for (auto&& slot = PolymorphicDowncast<Layer*>(m_Layer)->BeginInputSlots();
             slot != PolymorphicDowncast<Layer*>(m_Layer)->EndInputSlots();
             ++slot)
        {
            OutputSlot* parentLayerOutputSlot = slot->GetConnectedOutputSlot();
            ARMNN_ASSERT_MSG(parentLayerOutputSlot != nullptr, "The input slots must be connected here.");
            if (parentLayerOutputSlot)
            {
                Layer& parentLayer = parentLayerOutputSlot->GetOwningLayer();
                auto parentInfo = layerInfos.find(&parentLayer);
                if (parentInfo == layerInfos.end() ||
                        !m_Subgraph->IsMergedWith(parentInfo->second.m_Subgraph.get()))
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
                                       SubgraphView::IOutputSlots& outputSlots)
    {
        for (auto&& slot = PolymorphicDowncast<Layer*>(m_Layer)->BeginOutputSlots();
             slot != PolymorphicDowncast<Layer*>(m_Layer)->EndOutputSlots();
             ++slot)
        {
            for (InputSlot* childLayerInputSlot : slot->GetConnections())
            {
                Layer& childLayer = childLayerInputSlot->GetOwningLayer();
                auto childInfo = layerInfos.find(&childLayer);
                if (childInfo == layerInfos.end() ||
                        !m_Subgraph->IsMergedWith(childInfo->second.m_Subgraph.get()))
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

    IConnectableLayer* m_Layer;
    /// Which subgraph this layer has been assigned to. Only valid once m_IsProcessed is true.
    /// Two layers with different m_Subgraph pointers may in fact have been merged into the same subgraph -
    /// see the description of the PartialSubgraph class.
    std::shared_ptr<PartialSubgraph> m_Subgraph;
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
    Layer& layer = *PolymorphicDowncast<Layer*>(layerInfo.m_Layer);

    for (auto inputSlot : layer.GetInputSlots())
    {
        auto connectedInput = PolymorphicDowncast<OutputSlot*>(inputSlot.GetConnection());
        ARMNN_ASSERT_MSG(connectedInput, "Dangling input slot detected.");
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
    Layer& layer = *PolymorphicDowncast<Layer*>(layerInfo.m_Layer);

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
    // Check each input to see if we can attach ourselves to any of the subgraphs that have already been assigned.
    ForEachLayerInput(layerInfos, layerInfo, [&](LayerSelectionInfo& parentInfo)
    {
        // We can only attach ourselves to the subgraph from this input if there isn't a cut here.
        if (layerInfo.m_IsSelected == parentInfo.m_IsSelected)
        {
            // We also need to check that merging into this subgraph won't cause a dependency cycle between subgraphs.
            // This will be the case if the subgraph that we will become part of is already a dependency
            // of one of the subgraphs that are input to this layer, e.g:
            //
            //    0     |  The numbers (0, 1) are the subgraph IDs of each layer and we are looking at layer X.
            //   / \    |
            //  1   0   |  We can't merge X into subgraph 0, because the left-hand input already depends on subgraph 0.
            //   \ /    |  We can however merge X into subgraph 1.
            //    X     |
            //
            bool dependenciesOk = true;
            ForEachLayerInput(layerInfos, layerInfo, [&](LayerSelectionInfo& otherParentInfo)
            {
                // We call HasAntecedent() ~ n^2 times, where n is the number of inputs to this layer.
                // Hence it is important that this is efficient - see PartialSubgraph class description.
                if (otherParentInfo.m_Subgraph->HasAntecedent(parentInfo.m_Subgraph.get()))
                {
                    dependenciesOk = false;
                }
            });

            if (dependenciesOk)
            {
                // Merge into the subgraph of this input. If we have already been merged into another subgraph
                // (from another input of this layer), then merge both of them together.
                if (layerInfo.m_Subgraph == nullptr)
                {
                    layerInfo.m_Subgraph = parentInfo.m_Subgraph;
                }
                else
                {
                    // We call MergeWith() ~ n times, where n is the number of inputs to this layer.
                    // Therefore it does not need to be as performant as HasAntecedent().
                    layerInfo.m_Subgraph->MergeWith(parentInfo.m_Subgraph.get());
                }
            }
        }
    });

    // If we weren't able to merge into an existing subgraph then we need to make a new one
    if (layerInfo.m_Subgraph == nullptr)
    {
        layerInfo.m_Subgraph = std::make_shared<PartialSubgraph>();
    }

    // Record dependencies of the chosen subgraph based on the inputs of this layer.
    ForEachLayerInput(layerInfos, layerInfo, [&](LayerSelectionInfo& parentInfo)
    {
        // These functions are called ~n times, where n is the number of inputs to this layer.
        // Therefore it does not need to be as performant as HasAntecedent().
        if (!layerInfo.m_Subgraph->IsMergedWith(parentInfo.m_Subgraph.get()))
        {
            layerInfo.m_Subgraph->AddDirectAntecedent(parentInfo.m_Subgraph.get());
        }
    });
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
    const SubgraphView::IConnectableLayers& subgraphLayers = subgraph.GetIConnectableLayers();
    for (auto& layer : subgraphLayers)
    {

        auto emplaced = layerInfos.emplace(layer, LayerSelectionInfo{PolymorphicDowncast<Layer*>(layer), selector});
        LayerSelectionInfo& layerInfo = emplaced.first->second;

        // Start with Input type layers
        if (layerInfo.IsInputLayer())
        {
            processQueue.push(&layerInfo);
        }
    }

    const SubgraphView::IInputSlots& subgraphInputSlots = subgraph.GetIInputSlots();
    for (auto& inputSlot : subgraphInputSlots)
    {
        Layer& layer = PolymorphicDowncast<InputSlot*>(inputSlot)->GetOwningLayer();
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

    // Collect all selected layers keyed by subgraph representative into a map
    using SelectionInfoPtrs = std::vector<LayerSelectionInfo*>;
    std::map<PartialSubgraph*, SelectionInfoPtrs> splitMap;
    for (auto& info : layerInfos)
    {
        if (info.second.m_IsSelected)
        {
            auto it = splitMap.find(info.second.m_Subgraph->GetRepresentative());
            if (it == splitMap.end())
            {
                splitMap.insert(
                    std::make_pair(info.second.m_Subgraph->GetRepresentative(), SelectionInfoPtrs{&info.second}));
            }
            else
            {
                it->second.push_back(&info.second);
            }
        }
    }

    // Now each entry in splitMap represents a subgraph
    Subgraphs result;
    for (auto& splitGraph : splitMap)
    {
        SubgraphView::IInputSlots inputs;
        SubgraphView::IOutputSlots outputs;
        SubgraphView::IConnectableLayers layers;
        for (auto&& infoPtr : splitGraph.second)
        {
            infoPtr->CollectNonSelectedInputs(layerInfos, inputs);
            infoPtr->CollectNonSelectedOutputSlots(layerInfos, outputs);
            layers.push_back(infoPtr->m_Layer);
        }

        // Sort lists into deterministic order, not relying on pointer values which may be different on each execution.
        // This makes debugging the optimised graph much easier as subsequent stages can also be deterministic.
        std::sort(inputs.begin(), inputs.end(), [](const IInputSlot* a, const IInputSlot* b)
        {
            auto* castA = PolymorphicDowncast<const InputSlot*>(a);
            auto* castB = PolymorphicDowncast<const InputSlot*>(b);
            const LayerGuid guidA = castA->GetOwningLayer().GetGuid();
            const LayerGuid guidB = castB->GetOwningLayer().GetGuid();
            if (guidA < guidB)
            {
                return true;
            }
            else if (guidA == guidB)
            {
                return (castA->GetSlotIndex() < castB->GetSlotIndex());
            }
            return false;
        });
        std::sort(outputs.begin(), outputs.end(), [](const IOutputSlot* a, const IOutputSlot* b)
        {
            auto* castA = PolymorphicDowncast<const OutputSlot*>(a);
            auto* castB = PolymorphicDowncast<const OutputSlot*>(b);
            const LayerGuid guidA = castA->GetOwningLayer().GetGuid();
            const LayerGuid guidB = castB->GetOwningLayer().GetGuid();
            if (guidA < guidB)
            {
                return true;
            }
            else if (guidA == guidB)
            {
                return (a->CalculateIndexOnOwner() < b->CalculateIndexOnOwner());
            }
            return false;
        });
        layers.sort([](const IConnectableLayer* a, const IConnectableLayer* b) { return a->GetGuid() < b->GetGuid(); });

        // Create a new sub-graph with the new lists of input/output slots and layer
        result.emplace_back(std::make_unique<SubgraphView>(std::move(layers),
                                                           std::move(inputs),
                                                           std::move(outputs)));
    }

    return result;
}

} // namespace armnn
