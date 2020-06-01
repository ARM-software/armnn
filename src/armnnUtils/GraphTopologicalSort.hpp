//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <armnn/Optional.hpp>

#include <functional>
#include <map>
#include <stack>
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


template <typename TNodeId>
armnn::Optional<TNodeId> GetNextChild(TNodeId node,
                                      std::function<std::vector<TNodeId>(TNodeId)> getIncomingEdges,
                                      std::map<TNodeId, NodeState>& nodeStates)
{
    for (TNodeId childNode : getIncomingEdges(node))
    {
        if (nodeStates.find(childNode) == nodeStates.end())
        {
            return childNode;
        }
        else
        {
            if (nodeStates.find(childNode)->second == NodeState::Visiting)
            {
                return childNode;
            }
        }
    }

    return {};
}

template<typename TNodeId>
bool TopologicallySort(
    TNodeId initialNode,
    std::function<std::vector<TNodeId>(TNodeId)> getIncomingEdges,
    std::vector<TNodeId>& outSorted,
    std::map<TNodeId, NodeState>& nodeStates)
{
    std::stack<TNodeId> nodeStack;

    // If the node is never visited we should search it
    if (nodeStates.find(initialNode) == nodeStates.end())
    {
        nodeStack.push(initialNode);
    }

    while (!nodeStack.empty())
    {
        TNodeId current = nodeStack.top();

        nodeStates[current] = NodeState::Visiting;

        auto nextChildOfCurrent = GetNextChild(current, getIncomingEdges, nodeStates);

        if (nextChildOfCurrent)
        {
            TNodeId nextChild = nextChildOfCurrent.value();

            // If the child has not been searched, add to the stack and iterate over this node
            if (nodeStates.find(nextChild) == nodeStates.end())
            {
                nodeStack.push(nextChild);
                continue;
            }

            // If we re-encounter a node being visited there is a cycle
            if (nodeStates[nextChild] == NodeState::Visiting)
            {
                return false;
            }
        }

        nodeStack.pop();

        nodeStates[current] = NodeState::Visited;
        outSorted.push_back(current);
    }

    return true;
}

}

// Sorts a directed acyclic graph (DAG) into a flat list such that all inputs to a node are before the node itself.
// Returns true if successful or false if there is an error in the graph structure (e.g. it contains a cycle).
// The graph is defined entirely by the "getIncomingEdges" function which the user provides. For a given node,
// it must return the list of nodes which are required to come before it.
// "targetNodes" is the list of nodes where the search begins - i.e. the nodes that you want to evaluate.
// This is an iterative implementation based on https://en.wikipedia.org/wiki/Topological_sorting#Depth-first_search
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
        if (!TopologicallySort(targetNode, getIncomingEdges, outSorted, nodeStates))
        {
            return false;
        }
    }

    return true;
}

}