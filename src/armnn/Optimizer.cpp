//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include "Optimizer.hpp"
#include "Observable.hpp"
#include "optimizations/All.hpp"

namespace armnn
{

Optimizer::Optimizer()
{
}

void Optimizer::Pass(Graph& graph, const Optimizations& optimizations)
{
    ARMNN_SCOPED_PROFILING_EVENT(Compute::Undefined, "Optimizer_Pass");
    // Create observables to observe changes to the graph
    AddedLayerObservable addedLayerObservable(graph);
    ErasedLayerNamesObservable erasedLayerNamesObservable(graph);

    bool graphNeedsSorting = false;
    auto it = graph.TopologicalSort().end();

    // Calls TopologicalSort() for every iteration to re-order the list in case layers were added/removed.
    while (it != graph.TopologicalSort().begin())
    {
        --it;
        for (auto&& optimization : optimizations)
        {
            ARMNN_ASSERT(*it);
            optimization->Run(graph, **it);

            if ((*it)->IsOutputUnconnected())
            {
                auto next = std::next(graph.GetPosInGraph(**it));
                graph.EraseLayer(it);
                it = next;
                graphNeedsSorting = true;
            }

            // Add the names of erased layers as related layers to the new added layers
            for (auto& erasedLayerName : erasedLayerNamesObservable)
            {
                for (auto& addedLayer : addedLayerObservable)
                {
                    addedLayer->AddRelatedLayerName(erasedLayerName);
                }
            }

            erasedLayerNamesObservable.Clear();
            addedLayerObservable.Clear();

            if (graphNeedsSorting)
            {
                graphNeedsSorting = false;
                break;
            }
        }
    }
}

} // namespace armnn
