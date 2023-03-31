//
// Copyright © 2020,2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <cstdint>
#include <map>
#include <string>

namespace arm
{
namespace pipe
{

class ICounterMappings
{
public:
    virtual uint16_t GetGlobalId(uint16_t backendCounterId, const std::string& backendId) const = 0;
    virtual const std::pair<uint16_t, std::string>& GetBackendId(uint16_t globalCounterId) const = 0;
    virtual ~ICounterMappings() {}
};

class IRegisterCounterMapping
{
public:
    virtual void RegisterMapping(uint16_t globalCounterId,
                                 uint16_t backendCounterId,
                                 const std::string& backendId) = 0;
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
                         const std::string& backendId) override;
    void Reset() override;
    uint16_t GetGlobalId(uint16_t backendCounterId, const std::string& backendId) const override;
    const std::pair<uint16_t, std::string>& GetBackendId(uint16_t globalCounterId) const override;
private:
    std::map<uint16_t, std::pair<uint16_t, std::string>> m_GlobalCounterIdMap;
    std::map<std::pair<uint16_t, std::string>, uint16_t> m_BackendCounterIdMap;
};

}    // namespace pipe
}    // namespace arm
