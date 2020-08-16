//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <Graph.hpp>
#include <SubgraphView.hpp>
#include <SubgraphViewSelector.hpp>
#include <ResolveType.hpp>

#include <armnn/BackendRegistry.hpp>

#include <armnn/Types.hpp>
#include <backendsCommon/CpuTensorHandle.hpp>

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

// Utility template for comparing tensor elements
template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
bool Compare(T a, T b, float tolerance = 0.000001f)
{
    if (ArmnnType == armnn::DataType::Boolean)
    {
        // NOTE: Boolean is represented as uint8_t (with zero equals
        // false and everything else equals true), therefore values
        // need to be casted to bool before comparing them
        return static_cast<bool>(a) == static_cast<bool>(b);
    }

    // NOTE: All other types can be cast to float and compared with
    // a certain level of tolerance
    return std::fabs(static_cast<float>(a) - static_cast<float>(b)) <= tolerance;
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
