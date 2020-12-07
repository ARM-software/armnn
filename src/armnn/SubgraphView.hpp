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
/// The SubgraphView class represents a subgraph of a Graph.
/// The data it holds, points to data held by layers of the Graph, so the
/// the contents of the SubgraphView become invalid when the Layers are destroyed
/// or changed.
///
class SubgraphView final
{
public:
    template <typename Func>
    void ForEachLayer(Func func) const
    {
        for (auto it = m_Layers.begin(); it != m_Layers.end(); )
        {
             auto next = std::next(it);
             func(*it);
             it = next;
        }
    }

    using SubgraphViewPtr = std::unique_ptr<SubgraphView>;
    using InputSlots = std::vector<InputSlot*>;
    using OutputSlots = std::vector<OutputSlot*>;
    using Layers = std::list<Layer*>;
    using Iterator = Layers::iterator;
    using ConstIterator = Layers::const_iterator;

    /// Constructs a sub-graph from the entire given graph.
    explicit SubgraphView(Graph& graph);

    /// Constructs a sub-graph with the given arguments.
    SubgraphView(InputSlots&& inputs, OutputSlots&& outputs, Layers&& layers);

    /// Copy-constructor.
    SubgraphView(const SubgraphView& subgraph);

    /// Move-constructor.
    SubgraphView(SubgraphView&& subgraph);

    /// Constructs a sub-graph with only the given layer.
    SubgraphView(IConnectableLayer* layer);

    /// Move-assignment operator.
    SubgraphView& operator=(SubgraphView&& other);

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

    void Clear();

private:
    void CheckSubgraph();

    /// Arrange the order of layers topologically so that nodes can be visited in valid order
    void ArrangeBySortOrder();

    /// The list of pointers to the input slots of the parent graph.
    InputSlots m_InputSlots;

    /// The list of pointers to the output slots of the parent graph.
    OutputSlots m_OutputSlots;

    /// The list of pointers to the layers of the parent graph.
    Layers m_Layers;
};

///
/// Old SubGraph definition kept for backward compatibility only.
///
using SubGraph ARMNN_DEPRECATED_MSG("SubGraph is deprecated, use SubgraphView instead") = SubgraphView;

} // namespace armnn
