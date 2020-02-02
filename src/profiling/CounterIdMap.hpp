//
// Copyright © 2020 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include "armnn/BackendId.hpp"
#include <map>

namespace armnn
{
namespace profiling
{

class ICounterMappings
{
public:
    virtual uint16_t GetGlobalId(uint16_t backendCounterId, const armnn::BackendId& backendId) const = 0;
    virtual const std::pair<uint16_t, armnn::BackendId>& GetBackendId(uint16_t globalCounterId) const = 0;
    virtual ~ICounterMappings() {}
};

class IRegisterCounterMapping
{
public:
    virtual void RegisterMapping(uint16_t globalCounterId,
                                 uint16_t backendCounterId,
                                 const armnn::BackendId& backendId) = 0;
    virtual void Reset() = 0;
    virtual ~IRegisterCounterMapping() {}
};

class CounterIdMap : public ICounterMappings, public IRegisterCounterMapping
{

public:
    CounterIdMap() = default;
    virtual ~CounterIdMap() {}
    void RegisterMapping(uint16_t globalCounterId,
                         uint16_t backendCounterId,
                         const armnn::BackendId& backendId) override;
    void Reset() override;
    uint16_t GetGlobalId(uint16_t backendCounterId, const armnn::BackendId& backendId) const override;
    const std::pair<uint16_t, armnn::BackendId>& GetBackendId(uint16_t globalCounterId) const override;
private:
    std::map<uint16_t, std::pair<uint16_t, armnn::BackendId>> m_GlobalCounterIdMap;
    std::map<std::pair<uint16_t, armnn::BackendId>, uint16_t> m_BackendCounterIdMap;
};

}    // namespace profiling
}    // namespace armnn
