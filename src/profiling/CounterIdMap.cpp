//
// Copyright © 2020 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include "CounterIdMap.hpp"
#include "armnn/BackendId.hpp"
#include <armnn/Exceptions.hpp>
#include <map>

namespace armnn
{
namespace profiling
{

void CounterIdMap::RegisterMapping(uint16_t globalCounterId,
                                   uint16_t backendCounterId,
                                   const armnn::BackendId& backendId)
{
    std::pair<uint16_t, armnn::BackendId> backendIdPair(backendCounterId, backendId);
    m_GlobalCounterIdMap[globalCounterId] = backendIdPair;
    m_BackendCounterIdMap[backendIdPair] = globalCounterId;
}

void CounterIdMap::Reset()
{
    m_GlobalCounterIdMap.clear();
    m_BackendCounterIdMap.clear();
}

uint16_t CounterIdMap::GetGlobalId(uint16_t backendCounterId, const armnn::BackendId& backendId) const
{
    std::pair<uint16_t, armnn::BackendId> backendIdPair(backendCounterId, backendId);
    auto it = m_BackendCounterIdMap.find(backendIdPair);
    if (it == m_BackendCounterIdMap.end())
    {
        std::stringstream ss;
        ss << "No Backend Counter [" << backendIdPair.second << ":" << backendIdPair.first << "] registered";
        throw armnn::Exception(ss.str());
    }
    return it->second;
}

const std::pair<uint16_t, armnn::BackendId>& CounterIdMap::GetBackendId(uint16_t globalCounterId) const
{
    auto it = m_GlobalCounterIdMap.find(globalCounterId);
    if (it == m_GlobalCounterIdMap.end())
    {
        std::stringstream ss;
        ss << "No Global Counter ID [" << globalCounterId << "] registered";
        throw armnn::Exception(ss.str());
    }
    return it->second;
}

}    // namespace profiling
}    // namespace armnn
