//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "SubGraph.hpp"
#include "Graph.hpp"

#include <boost/numeric/conversion/cast.hpp>

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
        boost::ignore_unused(errorMessage);

        // Check if the item is valid
        BOOST_ASSERT_MSG(i, errorMessage.c_str());

        // Check if a duplicate has been found
        BOOST_ASSERT_MSG(duplicateSet.find(i) == duplicateSet.end(), errorMessage.c_str());

        duplicateSet.insert(i);
    });
}

} // anonymous namespace

SubGraph::SubGraph(Graph& graph)
    : m_InputSlots{}
    , m_OutputSlots{}
    , m_Layers(graph.begin(), graph.end())
    , m_ParentGraph(&graph)
{
    CheckSubGraph();
}

SubGraph::SubGraph(Graph* parentGraph, InputSlots&& inputs, OutputSlots&& outputs, Layers&& layers)
    : m_InputSlots{inputs}
    , m_OutputSlots{outputs}
    , m_Layers{layers}
    , m_ParentGraph(parentGraph)
{
    CheckSubGraph();
}

SubGraph::SubGraph(const SubGraph& referenceSubGraph, InputSlots&& inputs, OutputSlots&& outputs, Layers&& layers)
    : m_InputSlots{inputs}
    , m_OutputSlots{outputs}
    , m_Layers{layers}
    , m_ParentGraph(referenceSubGraph.m_ParentGraph)
{
    CheckSubGraph();
}

SubGraph::SubGraph(const SubGraph& subGraph)
    : m_InputSlots(subGraph.m_InputSlots.begin(), subGraph.m_InputSlots.end())
    , m_OutputSlots(subGraph.m_OutputSlots.begin(), subGraph.m_OutputSlots.end())
    , m_Layers(subGraph.m_Layers.begin(), subGraph.m_Layers.end())
    , m_ParentGraph(subGraph.m_ParentGraph)
{
    CheckSubGraph();
}

SubGraph::SubGraph(SubGraph&& subGraph)
    : m_InputSlots(std::move(subGraph.m_InputSlots))
    , m_OutputSlots(std::move(subGraph.m_OutputSlots))
    , m_Layers(std::move(subGraph.m_Layers))
    , m_ParentGraph(std::exchange(subGraph.m_ParentGraph, nullptr))
{
    CheckSubGraph();
}

SubGraph::SubGraph(const SubGraph& referenceSubGraph, IConnectableLayer* layer)
    : m_InputSlots{}
    , m_OutputSlots{}
    , m_Layers{boost::polymorphic_downcast<Layer*>(layer)}
    , m_ParentGraph(referenceSubGraph.m_ParentGraph)
{
    unsigned int numInputSlots = layer->GetNumInputSlots();
    m_InputSlots.resize(numInputSlots);
    for (unsigned int i = 0; i < numInputSlots; i++)
    {
        m_InputSlots.at(i) = boost::polymorphic_downcast<InputSlot*>(&(layer->GetInputSlot(i)));
    }

    unsigned int numOutputSlots = layer->GetNumOutputSlots();
    m_OutputSlots.resize(numOutputSlots);
    for (unsigned int i = 0; i < numOutputSlots; i++)
    {
        m_OutputSlots.at(i) = boost::polymorphic_downcast<OutputSlot*>(&(layer->GetOutputSlot(i)));
    }

    CheckSubGraph();
}

void SubGraph::CheckSubGraph()
{
    // Check that the sub-graph has a valid parent graph
    BOOST_ASSERT_MSG(m_ParentGraph, "Sub-graphs must have a parent graph");

    // Check for invalid or duplicate input slots
    AssertIfNullsOrDuplicates(m_InputSlots, "Sub-graphs cannot contain null or duplicate input slots");

    // Check for invalid or duplicate output slots
    AssertIfNullsOrDuplicates(m_OutputSlots, "Sub-graphs cannot contain null or duplicate output slots");

    // Check for invalid or duplicate layers
    AssertIfNullsOrDuplicates(m_Layers, "Sub-graphs cannot contain null or duplicate layers");

    // Check that all the layers of the sub-graph belong to the parent graph
    std::for_each(m_Layers.begin(), m_Layers.end(), [&](const Layer* l)
    {
        BOOST_ASSERT_MSG(std::find(m_ParentGraph->begin(), m_ParentGraph->end(), l) != m_ParentGraph->end(),
                         "Sub-graph layer is not a member of the parent graph");
    });
}

void SubGraph::Update(Graph &graph)
{
    m_InputSlots.clear();
    m_OutputSlots.clear();
    m_Layers.assign(graph.begin(), graph.end());
    m_ParentGraph = &graph;

    CheckSubGraph();
}

const SubGraph::InputSlots& SubGraph::GetInputSlots() const
{
    return m_InputSlots;
}

const SubGraph::OutputSlots& SubGraph::GetOutputSlots() const
{
    return m_OutputSlots;
}

const InputSlot* SubGraph::GetInputSlot(unsigned int index) const
{
    return m_InputSlots.at(index);
}

InputSlot* SubGraph::GetInputSlot(unsigned int index)
{
    return  m_InputSlots.at(index);
}

const OutputSlot* SubGraph::GetOutputSlot(unsigned int index) const
{
    return m_OutputSlots.at(index);
}

OutputSlot* SubGraph::GetOutputSlot(unsigned int index)
{
    return m_OutputSlots.at(index);
}

unsigned int SubGraph::GetNumInputSlots() const
{
    return boost::numeric_cast<unsigned int>(m_InputSlots.size());
}

unsigned int SubGraph::GetNumOutputSlots() const
{
    return boost::numeric_cast<unsigned int>(m_OutputSlots.size());
}

const SubGraph::Layers & SubGraph::GetLayers() const
{
    return m_Layers;
}

SubGraph::Layers::iterator SubGraph::begin()
{
    return m_Layers.begin();
}

SubGraph::Iterator SubGraph::end()
{
    return m_Layers.end();
}

SubGraph::ConstIterator SubGraph::begin() const
{
    return m_Layers.begin();
}

SubGraph::ConstIterator SubGraph::end() const
{
    return m_Layers.end();
}

SubGraph::ConstIterator SubGraph::cbegin() const
{
    return begin();
}

SubGraph::ConstIterator SubGraph::cend() const
{
    return end();
}

} // namespace armnn
