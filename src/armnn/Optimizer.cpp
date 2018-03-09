//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//
#include "Optimizer.hpp"
#include "optimizations/All.hpp"

namespace armnn
{

const Optimizer& Optimizer::Get()
{
    // Add optimizations here
    static optimizations::SquashEqualPermuteSiblings squashEqualPermuteSiblings;
    static optimizations::SquashEqualReshapeSiblings squashEqualReshapeSiblings;
    static optimizations::OptimizeInversePermutes optimizeInversePermutes;
    static optimizations::MovePermuteUp movePermuteUp;
    static optimizations::PermuteAsReshape permuteAsReshape;
    static optimizations::OptimizeConsecutiveReshapes optimizeConsecutiveReshapes;

    // Set optimizations in desired order
    static const Optimizer optimizer({
                                         &squashEqualPermuteSiblings,
                                         &squashEqualReshapeSiblings,
                                         &optimizeInversePermutes,
                                         &movePermuteUp,
                                         &permuteAsReshape,
                                         &optimizeConsecutiveReshapes,
                                     });

    return optimizer;
}

void Optimizer::Optimize(Graph& graph) const
{
    auto it = graph.TopologicalSort().end();
    // Call TopologicalSort() in every iteration to re-order the list in case layers where added/removed.
    while (it != graph.TopologicalSort().begin())
    {
        --it;
        for (auto&& optimization : m_Optimizations)
        {
            optimization->Run(graph, it);

            if ((*it)->IsOutputUnconnected())
            {
                it = graph.EraseLayer(it);
                break;
            }
        }
    }
}


} // namespace armnn
