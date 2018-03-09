//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//
#pragma once

#if ARMNN_PROFILING_ENABLED

#include "armnn/ArmNN.hpp"

#include <chrono>
#include <iosfwd>
#include <ctime>
#include <vector>
#include <stack>
#include <map>

namespace armnn
{

// Clock class that uses the same timestamp function as the Mali DDK
class monotonic_clock {
public:
    using duration = std::chrono::nanoseconds;
    using time_point = std::chrono::time_point<monotonic_clock, duration>;

    static std::chrono::time_point<monotonic_clock, std::chrono::nanoseconds> now() noexcept
    {
        timespec ts;
#if defined(CLOCK_MONOTONIC_RAW)
        clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
#else
        clock_gettime(CLOCK_MONOTONIC, &ts);
#endif
        return time_point(std::chrono::nanoseconds(ts.tv_sec*1000000000 + ts.tv_nsec));
    }
};

// Simple single-threaded profiler.
// Tracks events reported by BeginEvent()/EndEvent() and outputs detailed information and stats when
// Profiler::AnalyzeEventsAndWriteResults() is called.
class Profiler
{
public:
    // Marks the beginning of a user-defined event.
    // No attempt will be made to copy the name string: It must be known at compile time.
    void BeginEvent(Compute compute, const std::string name);

    // Marks the end of a user-defined event.
    void EndEvent(Compute compute);

    // Increments the event tag, allowing grouping of events in a user-defined manner (e.g. per inference).
    void UpdateEventTag() { ++m_EventTag; m_EventTagUpdated = true; }

    // Analyzes the tracked events and writes the results to the given output stream.
    // Please refer to the configuration variables in Profiling.cpp to customize the information written.
    void AnalyzeEventsAndWriteResults(std::ostream& outStream) const;

    // Accesses the singleton
    static Profiler& Get() { return s_Instance; }

    // Gets a string name for a given Compute device enum
    const char* GetEventComputeDevice(Compute compute) const;

    // Gets the color to render an event with, based on which device it denotes
    std::uint32_t GetEventColor(Compute compute) const;

    typedef monotonic_clock Clock;
    typedef std::chrono::time_point<Clock> TimePoint;

private:

    struct Marker
    {
        std::size_t m_Id;
        const std::string m_EventName;
        TimePoint m_TimeStamp;
        Compute m_ComputeDevice;
        std::uint32_t m_Tag;
    };

    struct ProfilingEvent
    {
        std::string m_Label;
        TimePoint m_StartTime;
        TimePoint m_StopTime;
        Compute m_Device;
        std::uint32_t m_Tag;

        double DurationMs() const
        {
            return std::chrono::duration<double>(m_StopTime - m_StartTime).count()*1000.0;
        }
    };

    struct ProfilingEventStats
    {
        double m_TotalMs;
        double m_MinMs;
        double m_MaxMs;
        std::uint32_t m_Count;
    };

    Profiler();
    ~Profiler();

    // Waits for a compute device to finish working to guarantee correct timings.
    // Currently used exclusively when emitting profiling events denoting GPU work.
    void WaitForDevice(Compute compute) const;

    void AnalyzeEventSequenceAndWriteResults(std::vector<ProfilingEvent>::const_iterator first,
                                             std::vector<ProfilingEvent>::const_iterator last,
                                             std::ostream& outStream) const;

    std::map<std::string, ProfilingEventStats> CalculateProfilingEventStats() const;

    std::stack<Marker> m_ObservedMarkers;
    std::vector<ProfilingEvent> m_EventSequence;
    std::uint32_t m_EventTag;
    std::uint32_t m_NestingLevel;
    bool m_EventTagUpdated;

    static Profiler s_Instance;
};

// Helper to easily add event markers to the codebase
class ScopedProfilingEvent
{
public:
    ScopedProfilingEvent(Compute compute, const std::string name)
        : m_Compute(compute)
    {
        Profiler::Get().BeginEvent(compute, name);
    }

    ~ScopedProfilingEvent()
    {
        Profiler::Get().EndEvent(m_Compute);
    }

private:
    armnn::Compute m_Compute;
};

} // namespace armnn

// Allows grouping events in an user-defined manner (e.g. per inference)
#define ARMNN_UPDATE_PROFILING_EVENT_TAG() armnn::Profiler::Get().UpdateEventTag();

// The event name must be known at compile time
#define ARMNN_SCOPED_PROFILING_EVENT(compute, name) armnn::ScopedProfilingEvent e_##__FILE__##__LINE__(compute, name);

#else

#define ARMNN_UPDATE_PROFILING_EVENT_TAG()
#define ARMNN_SCOPED_PROFILING_EVENT(compute, name)

#endif // ARMNN_PROFILING_ENABLED

