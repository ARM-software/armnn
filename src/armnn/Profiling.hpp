//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include "ProfilingEvent.hpp"

#include <armnn/utility/IgnoreUnused.hpp>
#include "armnn/IProfiler.hpp"

#include "WallClockTimer.hpp"

#include <chrono>
#include <iosfwd>
#include <ctime>
#include <vector>
#include <stack>
#include <map>

namespace armnn
{

// Simple single-threaded profiler.
// Tracks events reported by BeginEvent()/EndEvent() and outputs detailed information and stats when
// Profiler::AnalyzeEventsAndWriteResults() is called.
class Profiler final : public IProfiler
{
public:
    Profiler();
    ~Profiler();
    using InstrumentPtr = std::unique_ptr<Instrument>;

    // Marks the beginning of a user-defined event.
    // No attempt will be made to copy the name string: it must be known at compile time.
    Event* BeginEvent(const BackendId& backendId, const std::string& name, std::vector<InstrumentPtr>&& instruments);

    // Marks the end of a user-defined event.
    void EndEvent(Event* event);

    // Enables/disables profiling.
    void EnableProfiling(bool enableProfiling) override;

    // Checks if profiling is enabled.
    bool IsProfilingEnabled() override;

    // Increments the event tag, allowing grouping of events in a user-defined manner (e.g. per inference).
    void UpdateEventTag();

    // Analyzes the tracked events and writes the results to the given output stream.
    // Please refer to the configuration variables in Profiling.cpp to customize the information written.
    void AnalyzeEventsAndWriteResults(std::ostream& outStream) const override;

    // Print stats for events in JSON Format to the given output stream.
    void Print(std::ostream& outStream) const override;

    // Gets the color to render an event with, based on which device it denotes.
    uint32_t GetEventColor(const BackendId& backendId) const;

private:
    using EventPtr = std::unique_ptr<Event>;
    struct Marker
    {
        std::size_t m_Id;
    };

    struct ProfilingEventStats
    {
        double m_TotalMs;
        double m_MinMs;
        double m_MaxMs;
        uint32_t m_Count;
    };

    template<typename EventIterType>
    void AnalyzeEventSequenceAndWriteResults(EventIterType first, EventIterType last, std::ostream& outStream) const;

    std::map<std::string, ProfilingEventStats> CalculateProfilingEventStats() const;
    void PopulateInferences(std::vector<const Event*>& outInferences, int& outBaseLevel) const;
    void PopulateDescendants(std::map<const Event*, std::vector<const Event*>>& outDescendantsMap) const;

    std::stack<Event*> m_Parents;
    std::vector<EventPtr> m_EventSequence;
    bool m_ProfilingEnabled;

private:
    // Friend functions for unit testing, see ProfilerTests.cpp.
    friend size_t GetProfilerEventSequenceSize(armnn::Profiler* profiler);
};

// Singleton profiler manager.
// Keeps track of all the running profiler instances.
class ProfilerManager
{
public:
    // Register the given profiler as a thread local pointer.
    void RegisterProfiler(Profiler* profiler);

    // Gets the thread local pointer to the profiler.
    Profiler* GetProfiler();

    // Accesses the singleton.
    static ProfilerManager& GetInstance();

private:
    // The constructor is kept private so that other instances of this class (other that the singleton's)
    // can't be allocated.
    ProfilerManager() {}
};

// Helper to easily add event markers to the codebase.
class ScopedProfilingEvent
{
public:
    using InstrumentPtr = std::unique_ptr<Instrument>;

    template<typename... Args>
    ScopedProfilingEvent(const BackendId& backendId, const std::string& name, Args&&... args)
        : m_Event(nullptr)
        , m_Profiler(ProfilerManager::GetInstance().GetProfiler())
    {
        if (m_Profiler && m_Profiler->IsProfilingEnabled())
        {
            std::vector<InstrumentPtr> instruments(0);
            instruments.reserve(sizeof...(args)); //One allocation
            ConstructNextInVector(instruments, std::forward<Args>(args)...);
            m_Event = m_Profiler->BeginEvent(backendId, name, std::move(instruments));
        }
    }

    ~ScopedProfilingEvent()
    {
        if (m_Profiler && m_Event)
        {
            m_Profiler->EndEvent(m_Event);
        }
    }

private:

    void ConstructNextInVector(std::vector<InstrumentPtr>& instruments)
    {
        IgnoreUnused(instruments);
    }

    template<typename Arg, typename... Args>
    void ConstructNextInVector(std::vector<InstrumentPtr>& instruments, Arg&& arg, Args&&... args)
    {
        instruments.emplace_back(std::make_unique<Arg>(std::forward<Arg>(arg)));
        ConstructNextInVector(instruments, std::forward<Args>(args)...);
    }

    Event* m_Event;       ///< Event to track
    Profiler* m_Profiler; ///< Profiler used
};

} // namespace armnn

#define ARMNN_SCOPED_PROFILING_EVENT_WITH_INSTRUMENTS_UNIQUE_LOC_INNER(lineNumber, backendId, /*name,*/ ...) \
    armnn::ScopedProfilingEvent e_ ## lineNumber(backendId, /*name,*/ __VA_ARGS__);

#define ARMNN_SCOPED_PROFILING_EVENT_WITH_INSTRUMENTS_UNIQUE_LOC(lineNumber, backendId, /*name,*/ ...) \
    ARMNN_SCOPED_PROFILING_EVENT_WITH_INSTRUMENTS_UNIQUE_LOC_INNER(lineNumber, backendId, /*name,*/ __VA_ARGS__)

// The event name must be known at compile time i.e. if you are going to use this version of the macro
// in code the first argument you supply after the backendId must be the name.
// NOTE: need to pass the line number as an argument from here so by the time it gets to the UNIQUE_LOC_INNER
//       above it has expanded to a string and will concat (##) correctly with the 'e_' prefix to yield a
//       legal and unique variable name (so long as you don't use the macro twice on the same line).
//       The concat preprocessing operator (##) very unhelpfully will not expand macros see
//       https://gcc.gnu.org/onlinedocs/cpp/Concatenation.html for the gory details.
#define ARMNN_SCOPED_PROFILING_EVENT_WITH_INSTRUMENTS(backendId, /*name,*/ ...) \
    ARMNN_SCOPED_PROFILING_EVENT_WITH_INSTRUMENTS_UNIQUE_LOC(__LINE__,backendId, /*name,*/ __VA_ARGS__)

#define ARMNN_SCOPED_PROFILING_EVENT(backendId, name) \
    ARMNN_SCOPED_PROFILING_EVENT_WITH_INSTRUMENTS(backendId, name, armnn::WallClockTimer())
