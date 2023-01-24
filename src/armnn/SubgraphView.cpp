//
// Copyright © 2017, 2019-2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <armnn/backends/SubgraphView.hpp>

#include <Graph.hpp>

#include <armnn/utility/IgnoreUnused.hpp>
#include <armnn/utility/NumericCast.hpp>
#include <armnn/utility/PolymorphicDowncast.hpp>

#include <fmt/format.h>
#include <utility>

namespace armnn
{

namespace
{

template <class C>
void AssertIfNullsOrDuplicates(const C& container, const std::string& errorMessage)
{
    using T = typename C::value_type;
    std::unordered_set<T> duplicateSet;
    std::for_each(container.begin(), container.end(), [&duplicateSet, &errorMessage](const T& i)
    {
        // Ignore unused for release builds
        IgnoreUnused(errorMessage);

        // Check if the item is valid
        ARMNN_ASSERT_MSG(i, errorMessage.c_str());

        // Check if a duplicate has been found
        ARMNN_ASSERT_MSG(duplicateSet.find(i) == duplicateSet.end(), errorMessage.c_str());

        duplicateSet.insert(i);
    });
}

} // anonymous namespace

SubgraphView::SubgraphView(Graph& graph)
    : enable_shared_from_this()
    , m_InputSlots{}
    , m_OutputSlots{}
    , m_Layers(graph.begin(), graph.end())
    , m_IConnectableLayers(graph.begin(), graph.end())
{
    ArrangeBySortOrder();
    CheckSubgraph();
}

/// IConnectable Duplication to maintain backwards compatibility
SubgraphView::SubgraphView(InputSlots&& inputs, OutputSlots&& outputs, Layers&& layers)
    : enable_shared_from_this()
    , m_InputSlots{InputSlots{inputs.begin(), inputs.end()}}
    , m_IInputSlots{IInputSlots{inputs.begin(), inputs.end()}}
    , m_OutputSlots{OutputSlots{outputs.begin(), outputs.end()}}
    , m_IOutputSlots{IOutputSlots{outputs.begin(), outputs.end()}}
    , m_Layers(layers)
    , m_IConnectableLayers(IConnectableLayers{layers.begin(), layers.end()})
{
    ArrangeBySortOrder();
    CheckSubgraph();
}

/// IConnectable Duplication to maintain backwards compatibility
SubgraphView::SubgraphView(SubgraphView::IConnectableLayers&& layers,
                           SubgraphView::IInputSlots&& inputs,
                           SubgraphView::IOutputSlots&& outputs)
        : enable_shared_from_this()
        , m_IInputSlots{inputs}
        , m_IOutputSlots{outputs}
        , m_IConnectableLayers(IConnectableLayers{layers.begin(), layers.end()})
{
    // Cast from IConnectableLayer to Layer for backward compatibility
    auto f = [](IConnectableLayer* value)
    {
        return PolymorphicDowncast<Layer*>(value);
    };
    std::transform(layers.begin(), layers.end(), std::back_inserter(m_Layers), f);

    m_InputSlots.resize(inputs.size());
    m_IInputSlots.resize(inputs.size());
    for (unsigned int i = 0; i < inputs.size(); i++)
    {
        m_InputSlots.at(i) = PolymorphicDowncast<InputSlot*>(inputs[i]);
        m_IInputSlots.at(i) = inputs[i];
    }

    m_OutputSlots.resize(outputs.size());
    m_IOutputSlots.resize(outputs.size());
    for (unsigned int i = 0; i < outputs.size(); i++)
    {
        m_OutputSlots.at(i) = PolymorphicDowncast<OutputSlot*>(outputs[i]);
        m_IOutputSlots.at(i) = outputs[i];
    }

    ArrangeBySortOrder();
    CheckSubgraph();
}

/// IConnectable Duplication to maintain backwards compatibility
SubgraphView::SubgraphView(SubgraphView::IConnectableLayers&& layers,
                           SubgraphView::IInputSlots&& inputs,
                           SubgraphView::IOutputSlots&& outputs,
                           std::shared_ptr<SubgraphViewWorkingCopy> ptr)
        : enable_shared_from_this()
        , m_IInputSlots{inputs}
        , m_IOutputSlots{outputs}
        , m_IConnectableLayers(IConnectableLayers{layers.begin(), layers.end()})
        , p_WorkingCopyImpl(std::move(ptr))
{
    // Cast from IConnectableLayer to Layer for backward compatibility
    auto f = [](IConnectableLayer* value)
    {
        return PolymorphicDowncast<Layer*>(value);
    };
    std::transform(layers.begin(), layers.end(), std::back_inserter(m_Layers), f);

    m_InputSlots.resize(inputs.size());
    m_IInputSlots.resize(inputs.size());
    for (unsigned int i = 0; i < inputs.size(); i++)
    {
        m_InputSlots.at(i) = PolymorphicDowncast<InputSlot*>(inputs[i]);
        m_IInputSlots.at(i) = inputs[i];
    }

    m_OutputSlots.resize(outputs.size());
    m_IOutputSlots.resize(outputs.size());
    for (unsigned int i = 0; i < outputs.size(); i++)
    {
        m_OutputSlots.at(i) = PolymorphicDowncast<OutputSlot*>(outputs[i]);
        m_IOutputSlots.at(i) = outputs[i];
    }

    ArrangeBySortOrder();
    CheckSubgraph();
}

SubgraphView::SubgraphView(const SubgraphView& subgraph)
    : enable_shared_from_this()
    , m_InputSlots(subgraph.m_InputSlots.begin(), subgraph.m_InputSlots.end())
    , m_IInputSlots(subgraph.m_IInputSlots.begin(), subgraph.m_IInputSlots.end())
    , m_OutputSlots(subgraph.m_OutputSlots.begin(), subgraph.m_OutputSlots.end())
    , m_IOutputSlots(subgraph.m_IOutputSlots.begin(), subgraph.m_IOutputSlots.end())
    , m_Layers(subgraph.m_Layers.begin(), subgraph.m_Layers.end())
    , m_IConnectableLayers(IConnectableLayers{subgraph.m_IConnectableLayers.begin(),
                                              subgraph.m_IConnectableLayers.end()})
{
    ArrangeBySortOrder();
    CheckSubgraph();
}

SubgraphView::SubgraphView(SubgraphView&& subgraph)
    : enable_shared_from_this()
    , m_InputSlots(std::move(subgraph.m_InputSlots))
    , m_IInputSlots(std::move(subgraph.m_IInputSlots))
    , m_OutputSlots(std::move(subgraph.m_OutputSlots))
    , m_IOutputSlots(std::move(subgraph.m_IOutputSlots))
    , m_Layers(std::move(subgraph.m_Layers))
    , m_IConnectableLayers(std::move(subgraph.m_IConnectableLayers))
{
    ArrangeBySortOrder();
    CheckSubgraph();
}

SubgraphView::SubgraphView(IConnectableLayer* layer)
    : enable_shared_from_this()
    , m_Layers{PolymorphicDowncast<Layer*>(layer)}
    , m_IConnectableLayers{layer}
{
    unsigned int numInputSlots = layer->GetNumInputSlots();
    m_InputSlots.resize(numInputSlots);
    m_IInputSlots.resize(numInputSlots);
    for (unsigned int i = 0; i < numInputSlots; i++)
    {
        m_InputSlots.at(i) = PolymorphicDowncast<InputSlot*>(&(layer->GetInputSlot(i)));
        m_IInputSlots.at(i) = &(layer->GetInputSlot(i));
    }

    unsigned int numOutputSlots = layer->GetNumOutputSlots();
    m_OutputSlots.resize(numOutputSlots);
    m_IOutputSlots.resize(numOutputSlots);
    for (unsigned int i = 0; i < numOutputSlots; i++)
    {
        m_OutputSlots.at(i) = PolymorphicDowncast<OutputSlot*>(&(layer->GetOutputSlot(i)));
        m_IOutputSlots.at(i) = &(layer->GetOutputSlot(i));
    }

    CheckSubgraph();
}

SubgraphView& SubgraphView::operator=(SubgraphView&& other)
{
    m_InputSlots = std::move(other.m_InputSlots);
    m_IInputSlots = std::move(other.m_IInputSlots);
    m_OutputSlots = std::move(other.m_OutputSlots);
    m_IOutputSlots = std::move(other.m_IOutputSlots);
    m_Layers = std::move(other.m_Layers);
    m_IConnectableLayers = std::move(other.m_IConnectableLayers);

    CheckSubgraph();

    return *this;
}

void SubgraphView::CheckSubgraph()
{
    // Check for invalid or duplicate input slots
    AssertIfNullsOrDuplicates(m_InputSlots, "Sub-graphs cannot contain null or duplicate input slots");

    // Check for invalid or duplicate output slots
    AssertIfNullsOrDuplicates(m_OutputSlots, "Sub-graphs cannot contain null or duplicate output slots");

    // Check for invalid or duplicate layers
    AssertIfNullsOrDuplicates(m_Layers, "Sub-graphs cannot contain null or duplicate layers");

    // Check for invalid or duplicate input slots
    AssertIfNullsOrDuplicates(m_IInputSlots, "Sub-graphs cannot contain null or duplicate IInputSlots");

    // Check for invalid or duplicate output slots
    AssertIfNullsOrDuplicates(m_IOutputSlots, "Sub-graphs cannot contain null or duplicate IOutputSlots");

    // Check for invalid or duplicate layers
    AssertIfNullsOrDuplicates(m_IConnectableLayers,
                              "Sub-graphs cannot contain null or duplicate IConnectableLayers");
}

const SubgraphView::InputSlots& SubgraphView::GetInputSlots() const
{
    return m_InputSlots;
}

const SubgraphView::IInputSlots& SubgraphView::GetIInputSlots() const
{
    return m_IInputSlots;
}

const SubgraphView::OutputSlots& SubgraphView::GetOutputSlots() const
{
    return m_OutputSlots;
}

const SubgraphView::IOutputSlots& SubgraphView::GetIOutputSlots() const
{
    return m_IOutputSlots;
}

const InputSlot* SubgraphView::GetInputSlot(unsigned int index) const
{
    return m_InputSlots.at(index);
}

const IInputSlot* SubgraphView::GetIInputSlot(unsigned int index) const
{
    return m_IInputSlots.at(index);
}

InputSlot* SubgraphView::GetInputSlot(unsigned int index)
{
    return m_InputSlots.at(index);
}

IInputSlot* SubgraphView::GetIInputSlot(unsigned int index)
{
    return m_IInputSlots.at(index);
}

const OutputSlot* SubgraphView::GetOutputSlot(unsigned int index) const
{
    return m_OutputSlots.at(index);
}

const IOutputSlot* SubgraphView::GetIOutputSlot(unsigned int index) const
{
    return m_IOutputSlots.at(index);
}

OutputSlot* SubgraphView::GetOutputSlot(unsigned int index)
{
    return m_OutputSlots.at(index);
}

IOutputSlot* SubgraphView::GetIOutputSlot(unsigned int index)
{
    return m_IOutputSlots.at(index);
}

unsigned int SubgraphView::GetNumInputSlots() const
{
    return armnn::numeric_cast<unsigned int>(m_IInputSlots.size());
}

unsigned int SubgraphView::GetNumOutputSlots() const
{
    return armnn::numeric_cast<unsigned int>(m_IOutputSlots.size());
}

const SubgraphView::Layers& SubgraphView::GetLayers() const
{
    return m_Layers;
}

const SubgraphView::IConnectableLayers& SubgraphView::GetIConnectableLayers() const
{
    return m_IConnectableLayers;
}

SubgraphView::Iterator SubgraphView::begin()
{
    return m_Layers.begin();
}

SubgraphView::Iterator SubgraphView::end()
{
    return m_Layers.end();
}

// IConnectable Duplication to maintain backwards compatibility
SubgraphView::IConnectableLayerIterator SubgraphView::beginIConnectable()
{
    return m_IConnectableLayers.begin();
}

SubgraphView::IConnectableLayerIterator SubgraphView::endIConnectable()
{
    return m_IConnectableLayers.end();
}

SubgraphView::ConstIterator SubgraphView::begin() const
{
    return m_Layers.begin();
}

SubgraphView::ConstIterator SubgraphView::end() const
{
    return m_Layers.end();
}

// IConnectable Duplication to maintain backwards compatibility
SubgraphView::ConstIConnectableIterator SubgraphView::beginIConnectable() const
{
    return m_IConnectableLayers.begin();
}

SubgraphView::ConstIConnectableIterator SubgraphView::endIConnectable() const
{
    return m_IConnectableLayers.end();
}

SubgraphView::ConstIterator SubgraphView::cbegin() const
{
    // Ignore deprecated call as this is internal to SubgraphView
    ARMNN_NO_DEPRECATE_WARN_BEGIN
    return begin();
    ARMNN_NO_DEPRECATE_WARN_END
}

SubgraphView::ConstIterator SubgraphView::cend() const
{
    // Ignore deprecated call as this is internal to SubgraphView
    ARMNN_NO_DEPRECATE_WARN_BEGIN
    return end();
    ARMNN_NO_DEPRECATE_WARN_END
}

// IConnectable Duplication to maintain backwards compatibility
SubgraphView::ConstIConnectableIterator SubgraphView::cbeginIConnectable() const
{
    return beginIConnectable();
}

SubgraphView::ConstIConnectableIterator SubgraphView::cendIConnectable() const
{
    return endIConnectable();
}

void SubgraphView::Clear()
{
    m_InputSlots.clear();
    m_OutputSlots.clear();
    m_Layers.clear();

    m_IInputSlots.clear();
    m_IOutputSlots.clear();
    m_IConnectableLayers.clear();
}

void SubgraphView::ArrangeBySortOrder()
{
    using LayerList = std::list<Layer*>;
    auto compareLayerPriority = [](const LayerList::value_type& layerA, const LayerList::value_type& layerB)
        {
            return layerA->GetPriority() < layerB->GetPriority();
        };

    m_Layers.sort(compareLayerPriority);

    using IConnectableLayersList = std::list<IConnectableLayer*>;
    auto compareIConnectableLayerPriority = [](const IConnectableLayersList::value_type& layerA,
                                               const IConnectableLayersList::value_type& layerB)
        {
            return PolymorphicDowncast<Layer*>(layerA)->GetPriority() <
                   PolymorphicDowncast<Layer*>(layerB)->GetPriority();
        };

    m_IConnectableLayers.sort(compareIConnectableLayerPriority);
}

struct SubgraphView::SubgraphViewWorkingCopy
{
public:

    SubgraphViewWorkingCopy() = default;
    SubgraphViewWorkingCopy(Graph graph, std::shared_ptr<const SubgraphView> originalSubgraphView)
                            : m_Graph(graph)
                            , m_OriginalSubgraphView(originalSubgraphView)
    {};

    Graph m_Graph;
    std::shared_ptr<const SubgraphView> m_OriginalSubgraphView;

};

SubgraphView SubgraphView::GetWorkingCopy() const
{
    if (p_WorkingCopyImpl)
    {
        throw Exception("The SubgraphView calling GetWorkingCopy() is already a working copy. This function "
                        "should be called on original SubgraphView obtained from OptimizeSubgraphView()");
    }

    // Create a cut down SubgraphView with underlying graph containing only the relevant layers.
    // It needs its own underlying layers so that they can be replaced safely.
    auto ptr = std::make_shared<SubgraphViewWorkingCopy>(Graph(), shared_from_this());

    std::unordered_map<const IConnectableLayer*, IConnectableLayer*> originalToClonedLayerMap;
    std::list<armnn::IConnectableLayer*> originalSubgraphLayers = GetIConnectableLayers();

    for (auto&& originalLayer : originalSubgraphLayers)
    {
        Layer* const layer = PolymorphicDowncast<const Layer*>(originalLayer)->Clone(ptr->m_Graph);
        originalToClonedLayerMap.emplace(originalLayer, layer);
    }

    SubgraphView::IInputSlots workingCopyInputs;
    // Add IInputSlots to workingCopy
    for (auto originalSubgraphInputSlot : GetIInputSlots())
    {
        const IConnectableLayer& originalSubgraphLayer =
                PolymorphicDowncast<InputSlot*>(originalSubgraphInputSlot)->GetOwningLayer();

        auto* clonedLayer = originalToClonedLayerMap[&originalSubgraphLayer];

        workingCopyInputs.push_back(&clonedLayer->GetInputSlot(originalSubgraphInputSlot->GetSlotIndex()));
    }

    for (auto originalSubgraphLayer : originalSubgraphLayers)
    {
        IConnectableLayer* const clonedLayer = originalToClonedLayerMap[originalSubgraphLayer];

        // OutputLayers have no OutputSlots to be connected
        if (clonedLayer->GetType() != LayerType::Output)
        {
            // connect all cloned layers as per original subgraph
            for (unsigned int i = 0; i < clonedLayer->GetNumOutputSlots(); i++)
            {
                auto& originalOutputSlot = originalSubgraphLayer->GetOutputSlot(i);
                auto& clonedOutputSlot   = clonedLayer->GetOutputSlot(i);
                for (unsigned int j = 0; j < originalOutputSlot.GetNumConnections(); j++)
                {
                    // nextLayer is the layer with IInputSlot connected to IOutputSlot we are working on
                    const IConnectableLayer& nextLayerOnOriginalSubgraph =
                            originalOutputSlot.GetConnection(j)->GetOwningIConnectableLayer();

                    // Check the layer is in our map and so has a clonedLayer
                    if (originalToClonedLayerMap.find(&nextLayerOnOriginalSubgraph) != originalToClonedLayerMap.end())
                    {
                        auto* nextLayerOnClonedSubgraph = originalToClonedLayerMap[&nextLayerOnOriginalSubgraph];

                        auto index = PolymorphicDowncast<OutputSlot*>(
                                &originalOutputSlot)->GetConnection(j)->GetSlotIndex();

                        IInputSlot& inputSlot = nextLayerOnClonedSubgraph->GetInputSlot(index);

                        // Then make the connection
                        clonedOutputSlot.Connect(inputSlot);
                    }
                }
                // Copy the tensorInfo to the clonedOutputSlot
                clonedOutputSlot.SetTensorInfo(originalOutputSlot.GetTensorInfo());
            }
        }
    }

    SubgraphView::IOutputSlots workingCopyOutputs;

    // Add IOutputSlots to workingCopy
    for (auto outputSlot : GetIOutputSlots())
    {
        auto outputSlotIndex = outputSlot->CalculateIndexOnOwner();
        const IConnectableLayer& originalSubgraphLayer = outputSlot->GetOwningIConnectableLayer();

        // OutputLayers have no OutputSlots to be connected
        if (originalSubgraphLayer.GetType() != LayerType::Output)
        {
            IConnectableLayer* clonedLayer = originalToClonedLayerMap[&originalSubgraphLayer];

            // Add the OutputSlot of clonedLayer to WorkingCopy OutputSlots
            workingCopyOutputs.push_back(&clonedLayer->GetOutputSlot(outputSlotIndex));
        }
    }

    SubgraphView::IConnectableLayers workingCopyLayers;
    for (auto& pair : originalToClonedLayerMap)
    {
        workingCopyLayers.push_back(pair.second);
    }

    return {std::move(workingCopyLayers),
            std::move(workingCopyInputs),
            std::move(workingCopyOutputs),
            ptr};
}

void SubgraphView::SubstituteSubgraph(SubgraphView& subgraph, IConnectableLayer* substituteLayer)
{
    ARMNN_ASSERT(substituteLayer != nullptr);
    SubgraphView substituteSubgraph(substituteLayer);

    SubstituteSubgraph(subgraph, substituteSubgraph);
}

void SubgraphView::UpdateSubgraphViewSlotPointers(SubgraphView& patternSubgraph,
                                                  const SubgraphView& substituteSubgraph)
{
    std::vector<IInputSlot*>::iterator inputSlotPosition;
    // search for and erase any InputSlots that appear in the WorkingCopy that match those in the PatternSubgraph
    for (unsigned long idx = 0; idx < patternSubgraph.GetIInputSlots().size(); idx++)
    {
        IInputSlot *slot = patternSubgraph.GetIInputSlots()[idx];
        inputSlotPosition = std::find(m_IInputSlots.begin(), m_IInputSlots.end(), slot);
        if (inputSlotPosition != m_IInputSlots.end())
        {
            m_IInputSlots.erase(inputSlotPosition);

            // while here, with correct position, add in replacement InputSlot from the substituteSubgraph
            m_IInputSlots.insert(inputSlotPosition, substituteSubgraph.GetIInputSlots()[idx]);
        }
    }

    std::vector<IOutputSlot*>::iterator outputSlotPosition;
    // search for and erase any OutputSlots that appear in the WorkingCopy that match those in the PatternSubgraph
    for (unsigned long idx = 0; idx < patternSubgraph.GetIOutputSlots().size(); idx++)
    {
        IOutputSlot *slot = patternSubgraph.GetIOutputSlots()[idx];
        outputSlotPosition = std::find(m_IOutputSlots.begin(), m_IOutputSlots.end(), slot);
        if (outputSlotPosition != m_IOutputSlots.end())
        {
            m_IOutputSlots.erase(outputSlotPosition);

            // while here, with correct position, add in replacement OutputSlot from the substituteSubgraph
            m_IOutputSlots.insert(outputSlotPosition, substituteSubgraph.GetIOutputSlots()[idx]);
        }
    }
}

void SubgraphView::SubstituteSubgraph(SubgraphView& patternSubgraph, const SubgraphView& substituteSubgraph)
{
    if (!p_WorkingCopyImpl)
    {
        throw NullPointerException("The SubgraphView calling SubstituteSubgraphView is not a working copy. "
                                   "Call this function on SubgraphView returned from SubgraphView::GetWorkingCopy()");
    }

    auto numPatternInputs = patternSubgraph.GetIInputSlots().size();
    auto numSubInputs = substituteSubgraph.GetIInputSlots().size();
    if (numPatternInputs != numSubInputs)
    {
        throw armnn::InvalidArgumentException(
           fmt::format("Number of InputSlots on substitute SubgraphView ({}) must equal the number of"
                       " InputSlots on pattern SubgraphView ({})",
                       numSubInputs,
                       numPatternInputs));
    }

    auto numPatternOutputs = patternSubgraph.GetIOutputSlots().size();
    auto numSubOutputs = substituteSubgraph.GetIOutputSlots().size();
    if (numPatternOutputs != numSubOutputs)
    {
        throw armnn::InvalidArgumentException(
           fmt::format("Number of OutputSlots on substitute SubgraphView ({}) must equal the number of"
                       " OutputSlots on pattern SubgraphView ({})",
                       numSubOutputs,
                       numPatternOutputs));
    }

    // Add substitute layer to the Main graph i.e. graph in p_WorkingCopyImpl
    auto workingCopyGraph = &p_WorkingCopyImpl->m_Graph;
    substituteSubgraph.ForEachIConnectableLayer([workingCopyGraph](IConnectableLayer* iConnectableLayer)
                                                {
                                                    // Search WorkingCopy Graph for substituteLayer and add if missing
                                                    if (std::find(std::begin(workingCopyGraph->m_Layers),
                                                                  std::end(workingCopyGraph->m_Layers),
                                                                  iConnectableLayer) ==
                                                        std::end(workingCopyGraph->m_Layers))
                                                    {
                                                        auto layer = PolymorphicDowncast<Layer*>(iConnectableLayer);

                                                        layer->Reparent(*workingCopyGraph,
                                                                        (workingCopyGraph->m_Layers).end());

                                                        workingCopyGraph->m_LayersInOrder = false;
                                                    }
                                                });

    // Replace the old connections with connections to new layer
    workingCopyGraph->ReplaceSubgraphConnections(patternSubgraph, substituteSubgraph);

    // Update input/outputSlot pointers
    UpdateSubgraphViewSlotPointers(patternSubgraph, substituteSubgraph);

    // Delete the old layers.
    workingCopyGraph->EraseSubgraphLayers(patternSubgraph);

    // Sort
    workingCopyGraph->TopologicalSort();

    // Update SubgraphView layer pointers to match those of the internal WorkingCopy layer pointers
    m_IConnectableLayers = IConnectableLayers{ workingCopyGraph->m_Layers.begin(),
                                               workingCopyGraph->m_Layers.end() };
}

const SubgraphView::IInputSlots& SubgraphView::GetOriginalInputSlots() const
{
    if (!p_WorkingCopyImpl)
    {
        throw NullPointerException("The SubgraphView calling GetOriginalInputSlots is not a working copy. "
                                   "Call this function on SubgraphView returned from SubgraphView::GetWorkingCopy()");
    }
    if (!p_WorkingCopyImpl->m_OriginalSubgraphView)
    {
        throw NullPointerException("The working copy SubgraphView pointer to its original SubgraphView is null.");
    }
    return p_WorkingCopyImpl->m_OriginalSubgraphView->GetIInputSlots();
}
const SubgraphView::IOutputSlots& SubgraphView::GetOriginalOutputSlots() const
{
    if (!p_WorkingCopyImpl)
    {
        throw NullPointerException("The SubgraphView calling GetOriginalOutputSlots is not a working copy. "
                                   "Call this function on SubgraphView returned from SubgraphView::GetWorkingCopy()");
    }
    if (!p_WorkingCopyImpl->m_OriginalSubgraphView)
    {
        throw NullPointerException("The working copy SubgraphView pointer to its original SubgraphView is null.");
    }
    return p_WorkingCopyImpl->m_OriginalSubgraphView->GetIOutputSlots();
}


} // namespace armnn
