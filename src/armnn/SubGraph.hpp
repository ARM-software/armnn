//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "Layer.hpp"
#include "Graph.hpp"

#include <vector>
#include <list>

namespace armnn
{

///
/// The SubGraph class represents a subgraph of a Graph.
/// The data it holds, points to data held by layers of the Graph, so the
/// the contents of the SubGraph becomes invalid when the Layers are destroyed
/// or changed.
///
class SubGraph final
{
public:
    using SubGraphPtr = std::unique_ptr<SubGraph>;
    using InputSlots = std::vector<InputSlot*>;
    using OutputSlots = std::vector<OutputSlot*>;
    using Layers = std::list<Layer*>;
    using Iterator = Layers::iterator;
    using ConstIterator = Layers::const_iterator;

    /// Empty subgraphs are not allowed, they must at least have a parent graph.
    SubGraph() = delete;

    /// Constructs a sub-graph from the entire given graph.
    SubGraph(Graph& graph);

    /// Constructs a sub-graph with the given arguments and binds it to the specified parent graph.
    SubGraph(Graph* parentGraph, InputSlots&& inputs, OutputSlots&& outputs, Layers&& layers);

    /// Constructs a sub-graph with the given arguments and uses the specified sub-graph to get a reference
    /// to the parent graph.
    SubGraph(const SubGraph& referenceSubGraph, InputSlots&& inputs, OutputSlots&& outputs, Layers&& layers);

    /// Copy-constructor.
    SubGraph(const SubGraph& subGraph);

    /// Move-constructor.
    SubGraph(SubGraph&& subGraph);

    /// Constructs a sub-graph with only the given layer and uses the specified sub-graph to get a reference
    /// to the parent graph.
    SubGraph(const SubGraph& referenceSubGraph, IConnectableLayer* layer);

    /// Updates this sub-graph with the contents of the whole given graph.
    void Update(Graph& graph);

    /// Adds a new layer, of type LayerType, to the graph this sub-graph is a view of.
    template <typename LayerT, typename... Args>
    LayerT* AddLayer(Args&&... args) const;

    const InputSlots& GetInputSlots() const;
    const OutputSlots& GetOutputSlots() const;
    const Layers& GetLayers() const;

    const InputSlot* GetInputSlot(unsigned int index) const;
    InputSlot* GetInputSlot(unsigned int index);

    const OutputSlot* GetOutputSlot(unsigned int index) const;
    OutputSlot* GetOutputSlot(unsigned int index);

    unsigned int GetNumInputSlots() const;
    unsigned int GetNumOutputSlots() const;

    Iterator begin();
    Iterator end();

    ConstIterator begin() const;
    ConstIterator end() const;

    ConstIterator cbegin() const;
    ConstIterator cend() const;

private:
    void CheckSubGraph();

    InputSlots m_InputSlots;
    OutputSlots m_OutputSlots;
    Layers m_Layers;

    /// Pointer to the graph this sub-graph is a view of.
    Graph* m_ParentGraph;
};

template <typename LayerT, typename... Args>
LayerT* SubGraph::AddLayer(Args&&... args) const
{
    BOOST_ASSERT(m_ParentGraph);

    return m_ParentGraph->AddLayer<LayerT>(args...);
}

} // namespace armnn
