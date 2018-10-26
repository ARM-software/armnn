//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include "SubGraph.hpp"
#include <functional>
#include <memory>

namespace armnn
{

class Layer;
class Graph;

class SubGraphSelector final
{
public:
    using SubGraphPtr = std::unique_ptr<SubGraph>;
    using SubGraphs = std::vector<SubGraphPtr>;
    using LayerSelectorFunction = std::function<bool(const Layer&)>;

    /// Selects subgraphs of a graph based on the selector function
    /// and the algorithm. Since the SubGraphs returns modifiable pointers
    /// the input and output slots of the graph:
    ///  1) the graph cannot be const
    ///  2) the caller need to make sure that the SubGraphs lifetime is
    ///     shorter than the graph's
    static SubGraphs SelectSubGraphs(Graph& graph,
                                     const LayerSelectorFunction& selector);

private:
    // this is a utility class, don't construct or copy
    SubGraphSelector() = delete;
    SubGraphSelector(const SubGraphSelector&) = delete;
    SubGraphSelector & operator=(const SubGraphSelector&) = delete;
};

} // namespace armnn
