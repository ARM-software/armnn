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

    template <typename Func>
    void ForEachIConnectableLayer(Func func) const
    {
        for (auto it = m_IConnectableLayers.begin(); it != m_IConnectableLayers.end(); )
        {
             auto next = std::next(it);
             func(*it);
             it = next;
        }
    }

    using SubgraphViewPtr = std::unique_ptr<SubgraphView>;
    using InputSlots = std::vector<InputSlot*>;
    using IInputSlots = std::vector<IInputSlot*>;
    using OutputSlots = std::vector<OutputSlot*>;
    using IOutputSlots = std::vector<IOutputSlot*>;
    using Layers = std::list<Layer*>;
    using IConnectableLayers = std::list<IConnectableLayer*>;
    using Iterator = Layers::iterator;
    using IConnectableLayerIterator = IConnectableLayers::iterator;
    using ConstIterator = Layers::const_iterator;
    using ConstIConnectableIterator = IConnectableLayers::const_iterator;

    /// Constructs a sub-graph from the entire given graph.
    explicit SubgraphView(Graph& graph);

    /// Constructs a sub-graph with the given arguments.
    ARMNN_DEPRECATED_MSG_REMOVAL_DATE("This function has been deprecated, please use constructor with arguments: "
                                      "IConnectableLayers, IInputSlots and IOutputSlots", "22.08")
    SubgraphView(InputSlots&& inputs, OutputSlots&& outputs, Layers&& layers);

    /// Constructs a sub-graph with the given arguments.
    SubgraphView(IConnectableLayers&& layers, IInputSlots&& inputs, IOutputSlots&& outputs);

    /// Copy-constructor.
    SubgraphView(const SubgraphView& subgraph);

    /// Move-constructor.
    SubgraphView(SubgraphView&& subgraph);

    /// Constructs a sub-graph with only the given layer.
    SubgraphView(IConnectableLayer* layer);

    /// Move-assignment operator.
    SubgraphView& operator=(SubgraphView&& other);

    ARMNN_DEPRECATED_MSG_REMOVAL_DATE("This function has been deprecated, please use GetIInputSlots() returning"
                                      " public IInputSlots", "22.08")
    const InputSlots& GetInputSlots() const;
    const IInputSlots& GetIInputSlots() const;

    ARMNN_DEPRECATED_MSG_REMOVAL_DATE("This function has been deprecated, please use GetIOutputSlots() returning"
                                      " public IOutputSlots", "22.08")
    const OutputSlots& GetOutputSlots() const;
    const IOutputSlots& GetIOutputSlots() const;

    ARMNN_DEPRECATED_MSG_REMOVAL_DATE("This function has been deprecated, please use GetIConnectableLayers() "
                                      "returning public IConnectableLayers", "22.08")
    const Layers& GetLayers() const;
    const IConnectableLayers& GetIConnectableLayers() const;

    ARMNN_DEPRECATED_MSG_REMOVAL_DATE("This function has been deprecated, please use GetIInputSlot() returning public "
                                      "IInputSlot", "22.08")
    const InputSlot* GetInputSlot(unsigned int index) const;
    const IInputSlot* GetIInputSlot(unsigned int index) const;
    ARMNN_DEPRECATED_MSG_REMOVAL_DATE("This function has been deprecated, please use GetIInputSlot() returning public "
                                      "IInputSlot", "22.08")
    InputSlot* GetInputSlot(unsigned int index);
    IInputSlot* GetIInputSlot(unsigned int index);

    ARMNN_DEPRECATED_MSG_REMOVAL_DATE("This function has been deprecated, please use GetIOutputSlot() returning"
                                      " public IOutputSlot", "22.08")
    const OutputSlot* GetOutputSlot(unsigned int index) const;
    const IOutputSlot* GetIOutputSlot(unsigned int index) const;
    ARMNN_DEPRECATED_MSG_REMOVAL_DATE("This function has been deprecated, please use GetIOutputSlot() returning"
                                      " public IOutputSlot", "22.08")
    OutputSlot* GetOutputSlot(unsigned int index);
    IOutputSlot* GetIOutputSlot(unsigned int index);

    unsigned int GetNumInputSlots() const;
    unsigned int GetNumOutputSlots() const;

    ARMNN_DEPRECATED_MSG_CHANGE_DATE("This function is deprecated and will be changed to return an "
                                     "IConnectableLayerIterator, until that occurs in 23.02; please use "
                                     "beginIConnectable() returning public IConnectableLayerIterator", "23.02")
    Iterator begin();
    IConnectableLayerIterator beginIConnectable();
    ARMNN_DEPRECATED_MSG_CHANGE_DATE("This function is deprecated and will be changed to return an "
                                     "IConnectableLayerIterator, until that occurs in 23.02; please use "
                                     "endIConnectable() returning public IConnectableLayerIterator", "23.02")
    Iterator end();
    IConnectableLayerIterator endIConnectable();

    ARMNN_DEPRECATED_MSG_CHANGE_DATE("This function is deprecated and will be changed to return an "
                                     "ConstIConnectableIterator, until that occurs in 23.02; please use "
                                     "beginIConnectable() returning public ConstIConnectableIterator", "23.02")
    ConstIterator begin() const;
    ConstIConnectableIterator beginIConnectable() const;
    ARMNN_DEPRECATED_MSG_CHANGE_DATE("This function is deprecated and will be changed to return an "
                                     "ConstIConnectableIterator, until that occurs in 23.02; please use "
                                     "endIConnectable() returning public ConstIConnectableIterator", "23.02")
    ConstIterator end() const;
    ConstIConnectableIterator endIConnectable() const;

    ARMNN_DEPRECATED_MSG_CHANGE_DATE("This function is deprecated and will be changed to return an "
                                     "ConstIConnectableIterator, until that occurs in 23.02; please use "
                                     "cbeginIConnectable() returning public ConstIConnectableIterator", "23.02")
    ConstIterator cbegin() const;
    ConstIConnectableIterator cbeginIConnectable() const;
    ARMNN_DEPRECATED_MSG_CHANGE_DATE("This function is deprecated and will be changed to return an "
                                     "ConstIConnectableIterator, until that occurs in 23.02; please use "
                                     "cendIConnectable() returning public ConstIConnectableIterator", "23.02")
    ConstIterator cend() const;
    ConstIConnectableIterator cendIConnectable() const;

    void Clear();

private:
    void CheckSubgraph();

    /// Arrange the order of layers topologically so that nodes can be visited in valid order
    void ArrangeBySortOrder();

    /// The list of pointers to the input slots of the parent graph.
    InputSlots m_InputSlots;
    IInputSlots m_IInputSlots;

    /// The list of pointers to the output slots of the parent graph.
    OutputSlots m_OutputSlots;
    IOutputSlots m_IOutputSlots;

    /// The list of pointers to the layers of the parent graph.
    Layers m_Layers;
    IConnectableLayers m_IConnectableLayers;
};
} // namespace armnn
