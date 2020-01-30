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

class CounterIdMap
{

public:
    void RegisterMapping(uint16_t globalCounterId, uint16_t backendCounterId, const armnn::BackendId& backendId);
    uint16_t GetGlobalId(uint16_t backendCounterId, const armnn::BackendId& backendId);
    const std::pair<uint16_t, armnn::BackendId>& GetBackendId(uint16_t globalCounterId);
private:
    std::map<uint16_t, std::pair<uint16_t, armnn::BackendId>> m_GlobalCounterIdMap;
    std::map<std::pair<uint16_t, armnn::BackendId>, uint16_t> m_BackendCounterIdMap;
};

}    // namespace profiling
}    // namespace armnn