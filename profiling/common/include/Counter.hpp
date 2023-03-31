//
// Copyright © 2022,2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <cstdint>
#include <string>

namespace arm
{

namespace pipe
{

class Counter final
{
public:
    // Constructors
    Counter(const std::string& backendId,
            uint16_t           counterUid,
            uint16_t           maxCounterUid,
            uint16_t           counterClass,
            uint16_t           interpolation,
            double             multiplier,
            const std::string& name,
            const std::string& description,
            const std::string& units,
            uint16_t           deviceUid,
            uint16_t           counterSetUid)
        : m_BackendId(backendId)
        , m_Uid(counterUid)
        , m_MaxCounterUid(maxCounterUid)
        , m_Class(counterClass)
        , m_Interpolation(interpolation)
        , m_Multiplier(multiplier)
        , m_Name(name)
        , m_Description(description)
        , m_Units(units)
        , m_DeviceUid(deviceUid)
        , m_CounterSetUid(counterSetUid)
    {}

    // Fields
    std::string m_BackendId;
    uint16_t    m_Uid;
    uint16_t    m_MaxCounterUid;
    uint16_t    m_Class;
    uint16_t    m_Interpolation;
    double      m_Multiplier;
    std::string m_Name;
    std::string m_Description;
    std::string m_Units;      // Optional, leave empty if the counter does not need units

    // Connections
    uint16_t m_DeviceUid;     // Optional, set to zero if the counter is not associated with a device
    uint16_t m_CounterSetUid; // Optional, set to zero if the counter is not associated with a counter set
};

} // namespace pipe

} // namespace arm
