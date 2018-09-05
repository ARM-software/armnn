//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "Observable.hpp"

namespace armnn
{

void AddedLayerObservable::Update(Layer* graphLayer)
{
    m_ObservedObjects.emplace_back(graphLayer);
}

void ErasedLayerNamesObservable::Update(Layer* graphLayer)
{
    auto& relatedLayerNames = graphLayer->GetRelatedLayerNames();

    // If the erased layer has no related layers we take the erased layer's name
    // Otherwise we need to preserve the related layer names,
    // since we want to preserve the original graph's information
    if (relatedLayerNames.empty())
    {
        m_ObservedObjects.emplace_back(graphLayer->GetName());
    }
    else
    {
        for (auto& relatedLayerName : relatedLayerNames)
        {
            m_ObservedObjects.emplace_back(relatedLayerName);
        }
    }
}

}
