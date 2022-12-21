//
// Copyright © 2017, 2022-2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/Deprecated.hpp>

#include <vector>
#include <list>
#include <iterator>
#include <memory>

namespace armnn
{
class Graph;
class IConnectableLayer;
class IInputSlot;
class IOutputSlot;
class InputSlot;
class Layer;
class OutputSlot;

///
/// The SubgraphView class represents a subgraph of a Graph.
/// The data it holds, points to data held by layers of the Graph, so the
/// the contents of the SubgraphView become invalid when the Layers are destroyed
/// or changed.
///
class SubgraphView final : public std::enable_shared_from_this<SubgraphView>
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

    using SubgraphViewPtr = std::shared_ptr<SubgraphView>;
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
                                      "IConnectableLayers, IInputSlots and IOutputSlots", "23.08")
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
                                      " public IInputSlots", "23.08")
    const InputSlots& GetInputSlots() const;
    const IInputSlots& GetIInputSlots() const;

    ARMNN_DEPRECATED_MSG_REMOVAL_DATE("This function has been deprecated, please use GetIOutputSlots() returning"
                                      " public IOutputSlots", "23.08")
    const OutputSlots& GetOutputSlots() const;
    const IOutputSlots& GetIOutputSlots() const;

    ARMNN_DEPRECATED_MSG_REMOVAL_DATE("This function has been deprecated, please use GetIConnectableLayers() "
                                      "returning public IConnectableLayers", "23.08")
    const Layers& GetLayers() const;
    const IConnectableLayers& GetIConnectableLayers() const;

    ARMNN_DEPRECATED_MSG_REMOVAL_DATE("This function has been deprecated, please use GetIInputSlot() returning public "
                                      "IInputSlot", "23.08")
    const InputSlot* GetInputSlot(unsigned int index) const;
    const IInputSlot* GetIInputSlot(unsigned int index) const;
    ARMNN_DEPRECATED_MSG_REMOVAL_DATE("This function has been deprecated, please use GetIInputSlot() returning public "
                                      "IInputSlot", "23.08")
    InputSlot* GetInputSlot(unsigned int index);
    IInputSlot* GetIInputSlot(unsigned int index);

    ARMNN_DEPRECATED_MSG_REMOVAL_DATE("This function has been deprecated, please use GetIOutputSlot() returning"
                                      " public IOutputSlot", "23.08")
    const OutputSlot* GetOutputSlot(unsigned int index) const;
    const IOutputSlot* GetIOutputSlot(unsigned int index) const;
    ARMNN_DEPRECATED_MSG_REMOVAL_DATE("This function has been deprecated, please use GetIOutputSlot() returning"
                                      " public IOutputSlot", "23.08")
    OutputSlot* GetOutputSlot(unsigned int index);
    IOutputSlot* GetIOutputSlot(unsigned int index);

    unsigned int GetNumInputSlots() const;
    unsigned int GetNumOutputSlots() const;

    ARMNN_DEPRECATED_MSG_CHANGE_DATE("This function is deprecated and will be changed to return an "
                                     "IConnectableLayerIterator, until that occurs in 23.08; please use "
                                     "beginIConnectable() returning public IConnectableLayerIterator", "23.08")
    Iterator begin();
    IConnectableLayerIterator beginIConnectable();
    ARMNN_DEPRECATED_MSG_CHANGE_DATE("This function is deprecated and will be changed to return an "
                                     "IConnectableLayerIterator, until that occurs in 23.08; please use "
                                     "endIConnectable() returning public IConnectableLayerIterator", "23.08")
    Iterator end();
    IConnectableLayerIterator endIConnectable();

    ARMNN_DEPRECATED_MSG_CHANGE_DATE("This function is deprecated and will be changed to return an "
                                     "ConstIConnectableIterator, until that occurs in 23.08; please use "
                                     "beginIConnectable() returning public ConstIConnectableIterator", "23.08")
    ConstIterator begin() const;
    ConstIConnectableIterator beginIConnectable() const;
    ARMNN_DEPRECATED_MSG_CHANGE_DATE("This function is deprecated and will be changed to return an "
                                     "ConstIConnectableIterator, until that occurs in 23.08; please use "
                                     "endIConnectable() returning public ConstIConnectableIterator", "23.08")
    ConstIterator end() const;
    ConstIConnectableIterator endIConnectable() const;

    ARMNN_DEPRECATED_MSG_CHANGE_DATE("This function is deprecated and will be changed to return an "
                                     "ConstIConnectableIterator, until that occurs in 23.08; please use "
                                     "cbeginIConnectable() returning public ConstIConnectableIterator", "23.08")
    ConstIterator cbegin() const;
    ConstIConnectableIterator cbeginIConnectable() const;
    ARMNN_DEPRECATED_MSG_CHANGE_DATE("This function is deprecated and will be changed to return an "
                                     "ConstIConnectableIterator, until that occurs in 23.08; please use "
                                     "cendIConnectable() returning public ConstIConnectableIterator", "23.08")
    ConstIterator cend() const;
    ConstIConnectableIterator cendIConnectable() const;

    void Clear();

    /// This method returns a copy of the original SubgraphView provided by OptimizeSubgraphView with a separate
    /// underlying graph from the main ArmNN graph.
    /// Backend users should edit this working copy and then add it as a SubstitutionPair, along with original
    /// SubgraphView, to the OptimizationViews returned by OptimizeSubgraphView.
    /// ArmNN will then decide on whether or not to carry out Substitution of the two SubgraphViews.
    SubgraphView GetWorkingCopy() const;

    /// These methods should be called on a working copy subgraph created from GetWorkingCopy.
    /// They take a SubgraphView pattern to replace and the substitute layer or subgraphView to substitute in.
    void SubstituteSubgraph(SubgraphView&, IConnectableLayer*);
    void SubstituteSubgraph(SubgraphView&, const SubgraphView&);

    /// These methods should be called on a working copy subgraph created from GetWorkingCopy.
    /// They return pointers to the input and output Slots belonging to the original SubgraphView
    /// that the working copy was created from.
    /// This may be used to find the original TensorInfo of connected boundary OutputSlots.
    const IInputSlots& GetOriginalInputSlots() const;
    const IOutputSlots& GetOriginalOutputSlots() const;

private:
    struct SubgraphViewWorkingCopy;

    /// Constructs a sub-graph with the given arguments.
    SubgraphView(IConnectableLayers&& layers,
                 IInputSlots&& inputs,
                 IOutputSlots&& outputs,
                 std::shared_ptr<SubgraphViewWorkingCopy> ptr);

    void CheckSubgraph();

    /// Arrange the order of layers topologically so that nodes can be visited in valid order
    void ArrangeBySortOrder();

    /// Updates the IInputSlots and IOutputSlots variables assigned to a SubgraphView
    void UpdateSubgraphViewSlotPointers(SubgraphView&, const SubgraphView&);

    /// The list of pointers to the input slots of the parent graph.
    InputSlots m_InputSlots;
    IInputSlots m_IInputSlots;

    /// The list of pointers to the output slots of the parent graph.
    OutputSlots m_OutputSlots;
    IOutputSlots m_IOutputSlots;

    /// The list of pointers to the layers of the parent graph.
    Layers m_Layers;
    IConnectableLayers m_IConnectableLayers;

    /// Pointer to internal graph implementation. This stores a working copy of a graph, separate from the main
    /// ArmNN graph, for use by Backends so that they can edit it and add as a SubstitutionPair to OptimizationViews
    /// along with the original SubgraphView.
    /// ArmNN will then decide on whether or not to substitute in the provided SubgraphView working copy.
    std::shared_ptr<SubgraphViewWorkingCopy> p_WorkingCopyImpl;
};
} // namespace armnn
