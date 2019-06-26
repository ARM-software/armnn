//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <Graph.hpp>
#include <SubgraphView.hpp>
#include <SubgraphViewSelector.hpp>

#include <backendsCommon/CpuTensorHandle.hpp>
#include <backendsCommon/BackendRegistry.hpp>

#include <test/TestUtils.hpp>

#include <algorithm>

// Checks that two collections have the exact same contents (in any order)
// The given collections do not have to contain duplicates
// Cannot use std::sort here because std lists have their own std::list::sort method
template <typename CollectionType>
bool AreEqual(const CollectionType& lhs, const CollectionType& rhs)
{
    if (lhs.size() != rhs.size())
    {
        return false;
    }

    auto lhs_it = std::find_if(lhs.begin(), lhs.end(), [&rhs](auto& item)
    {
        return std::find(rhs.begin(), rhs.end(), item) == rhs.end();
    });

    return lhs_it == lhs.end();
}

// Checks that the given collection contains the specified item
template <typename CollectionType>
bool Contains(const CollectionType& collection, const typename CollectionType::value_type& item)
{
    return std::find(collection.begin(), collection.end(), item) != collection.end();
}

// Checks that the given map contains the specified key
template <typename MapType>
bool Contains(const MapType& map, const typename MapType::key_type& key)
{
    return map.find(key) != map.end();
}

template <typename ConvolutionLayer>
void SetWeightAndBias(ConvolutionLayer* layer, const armnn::TensorInfo& weightInfo, const armnn::TensorInfo& biasInfo)
{
    layer->m_Weight = std::make_unique<armnn::ScopedCpuTensorHandle>(weightInfo);
    layer->m_Bias   = std::make_unique<armnn::ScopedCpuTensorHandle>(biasInfo);

    layer->m_Weight->Allocate();
    layer->m_Bias->Allocate();
}

armnn::SubgraphView::InputSlots CreateInputsFrom(const std::vector<armnn::Layer*>& layers);

armnn::SubgraphView::OutputSlots CreateOutputsFrom(const std::vector<armnn::Layer*>& layers);

armnn::SubgraphView::SubgraphViewPtr CreateSubgraphViewFrom(armnn::SubgraphView::InputSlots&& inputs,
                                                            armnn::SubgraphView::OutputSlots&& outputs,
                                                            armnn::SubgraphView::Layers&& layers);

armnn::IBackendInternalUniquePtr CreateBackendObject(const armnn::BackendId& backendId);

armnn::TensorShape MakeTensorShape(unsigned int batches,
                                   unsigned int channels,
                                   unsigned int height,
                                   unsigned int width,
                                   armnn::DataLayout layout);