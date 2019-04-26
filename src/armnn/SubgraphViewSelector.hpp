//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include "SubgraphView.hpp"
#include <functional>
#include <memory>

namespace armnn
{

class Layer;
class Graph;

class SubgraphViewSelector final
{
public:
    using SubgraphViewPtr = std::unique_ptr<SubgraphView>;
    using Subgraphs = std::vector<SubgraphViewPtr>;
    using LayerSelectorFunction = std::function<bool(const Layer&)>;

    /// Selects subgraphs from a graph based on the selector function and the algorithm.
    /// Since the Subgraphs object returns modifiable pointers to the input and output slots of the graph:
    ///  1) the graph/sub-graph cannot be const
    ///  2) the caller needs to make sure that the Subgraphs lifetime is shorter than the parent graph's
    static Subgraphs SelectSubgraphs(Graph& graph, const LayerSelectorFunction& selector);
    static Subgraphs SelectSubgraphs(SubgraphView& subgraph, const LayerSelectorFunction& selector);

private:
    // this is a utility class, don't construct or copy
    SubgraphViewSelector() = delete;
    SubgraphViewSelector(const SubgraphViewSelector&) = delete;
    SubgraphViewSelector & operator=(const SubgraphViewSelector&) = delete;
};

} // namespace armnn
