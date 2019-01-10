//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "Profiling.hpp"
#include "ProfilingEvent.hpp"

namespace armnn
{
Event::Event(const std::string& eventName,
             Profiler* profiler,
             Event* parent,
             const BackendId backendId,
             std::vector<InstrumentPtr>&& instruments)
    : m_EventName(eventName)
    , m_Profiler(profiler)
    , m_Parent(parent)
    , m_BackendId(backendId)
    , m_Instruments(std::move(instruments))
{
}

Event::Event(Event&& other) noexcept
    : m_EventName(std::move(other.m_EventName))
    , m_Profiler(other.m_Profiler)
    , m_Parent(other.m_Parent)
    , m_BackendId(other.m_BackendId)
    , m_Instruments(std::move(other.m_Instruments))

{
}

Event::~Event() noexcept
{
}

void Event::Start()
{
    for (auto& instrument : m_Instruments)
    {
        instrument->Start();
    }
}

void Event::Stop()
{
    for (auto& instrument : m_Instruments)
    {
        instrument->Stop();
    }
}

const std::vector<Measurement> Event::GetMeasurements() const
{
    std::vector<Measurement> measurements;
    for (auto& instrument : m_Instruments)
    {
        for (auto& measurement : instrument->GetMeasurements())
        {
            measurements.emplace_back(std::move(measurement));
        }
    }
    return measurements;
}

const std::string& Event::GetName() const
{
    return m_EventName;
}

const Profiler* Event::GetProfiler() const
{
    return m_Profiler;
}

const Event* Event::GetParentEvent() const
{
    return m_Parent;
}

BackendId Event::GetBackendId() const
{
    return m_BackendId;
}

Event& Event::operator=(Event&& other) noexcept
{
    if (this == &other)
    {
        return *this;
    }

    m_EventName = other.m_EventName;
    m_Profiler = other.m_Profiler;
    m_Parent = other.m_Parent;
    m_BackendId = other.m_BackendId;
    other.m_Profiler = nullptr;
    other.m_Parent = nullptr;
    return *this;
}

} // namespace armnn
