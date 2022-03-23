//
// Copyright © 2020 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <client/include/CounterIdMap.hpp>

#include <common/include/ProfilingException.hpp>

#include <map>

namespace arm
{
namespace pipe
{

void CounterIdMap::RegisterMapping(uint16_t globalCounterId,
                                   uint16_t backendCounterId,
                                   const std::string& backendId)
{
    std::pair<uint16_t, std::string> backendIdPair(backendCounterId, backendId);
    m_GlobalCounterIdMap[globalCounterId] = backendIdPair;
    m_BackendCounterIdMap[backendIdPair] = globalCounterId;
}

void CounterIdMap::Reset()
{
    m_GlobalCounterIdMap.clear();
    m_BackendCounterIdMap.clear();
}

uint16_t CounterIdMap::GetGlobalId(uint16_t backendCounterId, const std::string& backendId) const
{
    std::pair<uint16_t, std::string> backendIdPair(backendCounterId, backendId);
    auto it = m_BackendCounterIdMap.find(backendIdPair);
    if (it == m_BackendCounterIdMap.end())
    {
        std::stringstream ss;
        ss << "No Backend Counter [" << backendIdPair.second << ":" << backendIdPair.first << "] registered";
        throw arm::pipe::ProfilingException(ss.str());
    }
    return it->second;
}

const std::pair<uint16_t, std::string>& CounterIdMap::GetBackendId(uint16_t globalCounterId) const
{
    auto it = m_GlobalCounterIdMap.find(globalCounterId);
    if (it == m_GlobalCounterIdMap.end())
    {
        std::stringstream ss;
        ss << "No Global Counter ID [" << globalCounterId << "] registered";
        throw arm::pipe::ProfilingException(ss.str());
    }
    return it->second;
}

}    // namespace pipe
}    // namespace arm
