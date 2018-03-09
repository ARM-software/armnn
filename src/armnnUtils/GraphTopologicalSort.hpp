//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//
#pragma once

#include <boost/assert.hpp>

#include <functional>
#include <map>
#include <vector>

namespace armnnUtils
{

namespace
{

enum class NodeState
{
    Visiting,
    Visited,
};

template<typename TNodeId>
bool Visit(
    TNodeId current,
    std::function<std::vector<TNodeId>(TNodeId)> getIncomingEdges,
    std::vector<TNodeId>& outSorted,
    std::map<TNodeId, NodeState>& nodeStates)
{
    auto currentStateIt = nodeStates.find(current);
    if (currentStateIt != nodeStates.end())
    {
        if (currentStateIt->second == NodeState::Visited)
        {
            return true;
        }
        if (currentStateIt->second == NodeState::Visiting)
        {
            return false;
        }
        else
        {
            BOOST_ASSERT(false);
        }
    }

    nodeStates[current] = NodeState::Visiting;

    for (TNodeId inputNode : getIncomingEdges(current))
    {
        Visit(inputNode, getIncomingEdges, outSorted, nodeStates);
    }

    nodeStates[current] = NodeState::Visited;

    outSorted.push_back(current);
    return true;
}

}

// Sorts an directed acyclic graph (DAG) into a flat list such that all inputs to a node are before the node itself.
// Returns true if successful or false if there is an error in the graph structure (e.g. it contains a cycle).
// The graph is defined entirely by the "getIncomingEdges" function which the user provides. For a given node,
// it must return the list of nodes which are required to come before it.
// "targetNodes" is the list of nodes where the search begins - i.e. the nodes that you want to evaluate.
// The implementation is based on https://en.wikipedia.org/wiki/Topological_sorting#Depth-first_search
template<typename TNodeId, typename TTargetNodes>
bool GraphTopologicalSort(
    const TTargetNodes& targetNodes,
    std::function<std::vector<TNodeId>(TNodeId)> getIncomingEdges,
    std::vector<TNodeId>& outSorted)
{
    outSorted.clear();
    std::map<TNodeId, NodeState> nodeStates;

    for (TNodeId targetNode : targetNodes)
    {
        if (!Visit(targetNode, getIncomingEdges, outSorted, nodeStates))
        {
            return false;
        }
    }

    return true;
}

}