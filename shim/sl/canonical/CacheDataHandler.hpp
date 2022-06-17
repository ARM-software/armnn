//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <vector>
#include <unordered_map>

#include <nnapi/Types.h>

namespace armnn_driver
{

class CacheHandle
{
public:
    CacheHandle(const android::nn::CacheToken token, const size_t cacheSize)
    : m_CacheToken(token), m_CacheSize(cacheSize) {}

    ~CacheHandle() {};

    android::nn::CacheToken GetToken() const
    {
        return m_CacheToken;
    }

    size_t GetCacheSize() const
    {
        return m_CacheSize;
    }

private:
    const android::nn::CacheToken m_CacheToken;
    const size_t m_CacheSize;
};

class CacheDataHandler
{
public:
    CacheDataHandler() {}
    ~CacheDataHandler() {}

    void Register(const android::nn::CacheToken token, const size_t hashValue, const size_t cacheSize);

    bool Validate(const android::nn::CacheToken token, const size_t hashValue, const size_t cacheSize) const;

    size_t Hash(std::vector<uint8_t>& cacheData);

    size_t GetCacheSize(android::nn::CacheToken token);

    void Clear();

private:
    CacheDataHandler(const CacheDataHandler&) = delete;
    CacheDataHandler& operator=(const CacheDataHandler&) = delete;

    std::unordered_map<size_t, CacheHandle> m_CacheDataMap;
};

CacheDataHandler& CacheDataHandlerInstance();

} // armnn_driver
