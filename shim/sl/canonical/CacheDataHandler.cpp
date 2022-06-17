//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "CacheDataHandler.hpp"

#include <log/log.h>

namespace armnn_driver
{

CacheDataHandler& CacheDataHandlerInstance()
{
    static CacheDataHandler instance;
    return instance;
}

void CacheDataHandler::Register(const android::nn::CacheToken token, const size_t hashValue, const size_t cacheSize)
{
    if (!m_CacheDataMap.empty()
            && m_CacheDataMap.find(hashValue) != m_CacheDataMap.end()
            && m_CacheDataMap.at(hashValue).GetToken() == token
            && m_CacheDataMap.at(hashValue).GetCacheSize() == cacheSize)
    {
        return;
    }
    CacheHandle cacheHandle(token, cacheSize);
    m_CacheDataMap.insert({hashValue, cacheHandle});
}

bool CacheDataHandler::Validate(const android::nn::CacheToken token,
                                const size_t hashValue,
                                const size_t cacheSize) const
{
    return (!m_CacheDataMap.empty()
            && m_CacheDataMap.find(hashValue) != m_CacheDataMap.end()
            && m_CacheDataMap.at(hashValue).GetToken() == token
            && m_CacheDataMap.at(hashValue).GetCacheSize() == cacheSize);
}

size_t CacheDataHandler::Hash(std::vector<uint8_t>& cacheData)
{
    std::size_t hash = cacheData.size();
    for (auto& i : cacheData)
    {
        hash = ((hash << 5) - hash) + i;
    }
    return hash;
}

size_t CacheDataHandler::GetCacheSize(android::nn::CacheToken token)
{
    for (auto i = m_CacheDataMap.begin(); i != m_CacheDataMap.end(); ++i)
    {
        if (i->second.GetToken() == token)
        {
            return i->second.GetCacheSize();
        }
    }
    return 0;
}

void CacheDataHandler::Clear()
{
    m_CacheDataMap.clear();
}

} // armnn_driver
