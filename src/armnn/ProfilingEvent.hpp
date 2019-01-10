//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <stack>
#include <vector>
#include <chrono>
#include <memory>
#include "Instrument.hpp"
#include "armnn/Types.hpp"

namespace armnn
{

/// Forward declaration
class Profiler;

/// Event class records measurements reported by BeginEvent()/EndEvent() and returns measurements when
/// Event::GetMeasurements() is called.
class Event
{
public:
    using InstrumentPtr = std::unique_ptr<Instrument>;
    using Instruments = std::vector<InstrumentPtr>;

    Event(const std::string& eventName,
          Profiler* profiler,
          Event* parent,
          const BackendId backendId,
          std::vector<InstrumentPtr>&& instrument);

    Event(const Event& other) = delete;

    /// Move Constructor
    Event(Event&& other) noexcept;

    /// Destructor
    ~Event() noexcept;

    /// Start the Event
    void Start();

    /// Stop the Event
    void Stop();

    /// Get the recorded measurements calculated between Start() and Stop()
    /// \return Recorded measurements of the event
    const std::vector<Measurement> GetMeasurements() const;

    /// Get the name of the event
    /// \return Name of the event
    const std::string& GetName() const;

    /// Get the pointer of the profiler associated with this event
    /// \return Pointer of the profiler associated with this event
    const Profiler* GetProfiler() const;

    /// Get the pointer of the parent event
    /// \return Pointer of the parent event
    const Event* GetParentEvent() const;

    /// Get the backend id of the event
    /// \return Backend id of the event
    BackendId GetBackendId() const;

    /// Assignment operator
    Event& operator=(const Event& other) = delete;

    /// Move Assignment operator
    Event& operator=(Event&& other) noexcept;

private:
    /// Name of the event
    std::string m_EventName;

    /// Stored associated profiler
    Profiler* m_Profiler;

    /// Stores optional parent event
    Event* m_Parent;

    /// Backend id
    BackendId m_BackendId;

    /// Instruments to use
    Instruments m_Instruments;
};

} // namespace armnn
