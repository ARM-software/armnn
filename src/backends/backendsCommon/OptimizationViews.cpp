//
// Copyright © 2017,2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <armnn/backends/OptimizationViews.hpp>

namespace armnn
{

bool OptimizationViews::Validate(const armnn::SubgraphView& originalSubgraph) const
{
    //This needs to verify that:
    // 1) the sum of m_SuccesfulOptimizations & m_FailedOptimizations & m_UntouchedSubgraphs contains subgraphviews
    //    which cover the entire space of the originalSubgraph.
    // 2) Each SubstitutionPair contains matching inputs and outputs
    bool valid = true;

    // Create a copy of the layer list from the original subgraph and sort it
    SubgraphView::IConnectableLayers originalLayers = originalSubgraph.GetIConnectableLayers();
    originalLayers.sort();

    // Create a new list based on the sum of all the subgraphs and sort it
    SubgraphView::IConnectableLayers countedLayers;
    for (auto& failed : m_FailedOptimizations)
    {
        countedLayers.insert(countedLayers.end(),
                             failed.GetIConnectableLayers().begin(),
                             failed.GetIConnectableLayers().end());
    }
    for (auto& untouched : m_UntouchedSubgraphs)
    {
        countedLayers.insert(countedLayers.end(),
                             untouched.GetIConnectableLayers().begin(),
                             untouched.GetIConnectableLayers().end());
    }
    for (auto& successful : m_SuccesfulOptimizations)
    {
        countedLayers.insert(countedLayers.end(),
                             successful.m_SubstitutableSubgraph.GetIConnectableLayers().begin(),
                             successful.m_SubstitutableSubgraph.GetIConnectableLayers().end());
    }
    countedLayers.sort();

    // Compare the two lists to make sure they match
    valid &= originalLayers.size() == countedLayers.size();

    auto oIt = originalLayers.begin();
    auto cIt = countedLayers.begin();
    for (size_t i=0; i < originalLayers.size() && valid; ++i, ++oIt, ++cIt)
    {
        valid &= (*oIt == *cIt);
    }

    // Compare the substitution subgraphs to ensure they are compatible
    if (valid)
    {
        for (auto& substitution : m_SuccesfulOptimizations)
        {
            bool validSubstitution = true;
            const SubgraphView& replacement = substitution.m_ReplacementSubgraph;
            const SubgraphView& old = substitution.m_SubstitutableSubgraph;
            validSubstitution &= replacement.GetIInputSlots().size() == old.GetIInputSlots().size();
            validSubstitution &= replacement.GetIOutputSlots().size() == old.GetIOutputSlots().size();
            valid &= validSubstitution;
        }
    }
    return valid;
}
} //namespace armnn
