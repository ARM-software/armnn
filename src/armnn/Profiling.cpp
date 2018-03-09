//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//
#include "Profiling.hpp"

#if ARMNN_PROFILING_ENABLED

#if ARMNN_STREAMLINE_ENABLED
#include <streamline_annotate.h>
#endif

#if ARMCOMPUTECL_ENABLED
#include <arm_compute/runtime/CL/CLFunctions.h>
#endif

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <map>
#include <stack>
#include <boost/algorithm/string.hpp>

namespace armnn
{

// Controls the amount of memory initially allocated to store profiling events.
// If chosen carefully, the profiling system will not make any additional allocations, thus minimizing its impact on
// measured times.
constexpr std::size_t g_ProfilingEventCountHint = 1024;

// Whether profiling reports should include the sequence of events together with their timings.
constexpr bool g_WriteProfilingEventSequence = true;

// Whether profiling reports should also report detailed information on events grouped by tag.
// This is used to group stats per inference (see usage of ARMNN_UPDATE_PROFILING_EVENT_TAG in
// Runtime::EnqueueWorkload). This can spam the output stream, so use carefully (or adapt
// the code to just output information for a tag of interest).
constexpr bool g_AggregateProfilingEventsByTag = false;

// Whether a call to Profiler::AnalyzeEventsAndWriteResults() will be made when the Profiler
// singleton is destroyed. It can be convenient for local tests.
constexpr bool g_WriteReportToStdOutOnProfilerDestruction = true;

// Whether events denoting operations running on the GPU should force a sync before/after the event.
// This is hardcoded to true for now as the profiling timings are not very useful without it.
constexpr bool g_ProfilingForceGpuSync = true;

std::map<std::string, Profiler::ProfilingEventStats> Profiler::CalculateProfilingEventStats() const
{
    std::map<std::string, ProfilingEventStats> nameToStatsMap;

    for (auto&& event : m_EventSequence)
    {
        auto mapIter = nameToStatsMap.find(event.m_Label);
        if (mapIter != nameToStatsMap.end())
        {
            ProfilingEventStats& stats = mapIter->second;
            stats.m_TotalMs += event.DurationMs();
            stats.m_MinMs = std::min(stats.m_MinMs, event.DurationMs());
            stats.m_MaxMs = std::max(stats.m_MaxMs, event.DurationMs());
            ++stats.m_Count;
        }
        else
        {
            ProfilingEventStats stats;
            stats.m_TotalMs = event.DurationMs();
            stats.m_MinMs = event.DurationMs();
            stats.m_MaxMs = event.DurationMs();
            stats.m_Count = 1;

            nameToStatsMap[event.m_Label] = stats;
        }
    }

    return nameToStatsMap;
}

void Profiler::AnalyzeEventSequenceAndWriteResults(std::vector<ProfilingEvent>::const_iterator first,
                                                   std::vector<ProfilingEvent>::const_iterator last,
                                                   std::ostream& outStream) const
{
    // Output event sequence, if needed
    if (g_WriteProfilingEventSequence)
    {
        // Make sure timestamps are output with 6 decimals, and save old settings
        std::streamsize oldPrecision = outStream.precision();
        outStream.precision(6);
        std::ios_base::fmtflags oldFlags = outStream.flags();
        outStream.setf(std::ios::fixed);
        // Output fields
        outStream << "Event Sequence - Name | Duration (ms) | Start (ms) | Stop (ms) | Device" << std::endl;
        for (auto event = first; event != last; ++event)
        {
            std::chrono::duration<double, std::milli> startTimeMs = event->m_StartTime.time_since_epoch();
            std::chrono::duration<double, std::milli> stopTimeMs = event->m_StopTime.time_since_epoch();

            outStream << std::setw(50) << event->m_Label << " "
                << std::setw(20) << event->DurationMs()
                << std::setw(20) << startTimeMs.count()
                << std::setw(20) << stopTimeMs.count()
                << std::setw(20) << Profiler::Get().GetEventComputeDevice(event->m_Device)
                << std::endl;
        }
        outStream << std::endl;
        // Restore previous precision settings
        outStream.flags(oldFlags);
        outStream.precision(oldPrecision);
    }

    // Aggregate results per event name
    std::map<std::string, ProfilingEventStats> nameToStatsMap = CalculateProfilingEventStats();

    // Output aggregated stats
    outStream << "Event Stats - Name | Avg (ms) | Min (ms) | Max (ms) | Total (ms) | Count" << std::endl;
    for (const auto& pair : nameToStatsMap)
    {
        const std::string& eventLabel = pair.first;
        const ProfilingEventStats& eventStats = pair.second;
        const double avgMs = eventStats.m_TotalMs / double(eventStats.m_Count);

        outStream << "\t" << std::setw(50) << eventLabel << " " << std::setw(9) << avgMs << " "
            << std::setw(9) << eventStats.m_MinMs << " " << std::setw(9) << eventStats.m_MaxMs << " "
            << std::setw(9) << eventStats.m_TotalMs << " " << std::setw(9) << eventStats.m_Count << std::endl;
    }
    outStream << std::endl;
}

Profiler Profiler::s_Instance;

Profiler::Profiler()
    : m_EventTag(0)
    , m_NestingLevel(0)
    , m_EventTagUpdated(false)
{
    m_EventSequence.reserve(g_ProfilingEventCountHint);

#if ARMNN_STREAMLINE_ENABLED
    // Initialise streamline annotations
    ANNOTATE_SETUP;
#endif
}

Profiler::~Profiler()
{
    if (g_WriteReportToStdOutOnProfilerDestruction)
    {
        AnalyzeEventsAndWriteResults(std::cout);
    }
}

void Profiler::BeginEvent(Compute compute, const std::string label)
{
    // We need to sync just before the begin event to not include time before the period we want to time.
    WaitForDevice(compute);

    const TimePoint timeStamp = Clock::now();
    m_ObservedMarkers.emplace(Marker{m_EventSequence.size(), label, timeStamp, compute, m_EventTag});
    m_EventSequence.emplace_back();

#if ARMNN_STREAMLINE_ENABLED
    ANNOTATE_CHANNEL_COLOR(m_NestingLevel, GetEventColor(compute), label.c_str());
#endif

    m_NestingLevel++;
}

void Profiler::EndEvent(Compute compute)
{
    // We need to sync just before the end event to include all the time of the timed period.
    WaitForDevice(compute);

    const Marker& marker = m_ObservedMarkers.top();

    const TimePoint startTime = marker.m_TimeStamp;
    const TimePoint stopTime = Clock::now();

    m_EventSequence[marker.m_Id] = {std::move(marker.m_EventName),
                                    startTime,
                                    stopTime,
                                    marker.m_ComputeDevice,
                                    marker.m_Tag};

    m_ObservedMarkers.pop();

#if ARMNN_STREAMLINE_ENABLED
    ANNOTATE_CHANNEL_END(m_NestingLevel);
#endif

    m_NestingLevel--;
}

void Profiler::AnalyzeEventsAndWriteResults(std::ostream& outStream) const
{
    // Stack should be empty now.
    const bool saneMarkerSequence = m_ObservedMarkers.empty();

    // Abort if the sequence of markers was found to have incorrect information:
    // The stats cannot be trusted.
    if (!saneMarkerSequence)
    {
        outStream << "Cannot write profiling stats. "
            "Unexpected errors were found when analyzing the sequence of logged events, which may lead to plainly "
            "wrong stats. The profiling system may contain implementation issues or could have been used in an "
            "unsafe manner." << std::endl;
        return;
    }

    // Analyze the full sequence of events
    AnalyzeEventSequenceAndWriteResults(m_EventSequence.begin(), m_EventSequence.end(), outStream);

    // Aggregate events by tag if requested (spams the output stream if done for all tags)
    if (m_EventTagUpdated && g_AggregateProfilingEventsByTag)
    {
        outStream << std::endl;
        outStream << "***" << std::endl;
        outStream << "*** Per Tag Stats" << std::endl;
        outStream << "***" << std::endl;
        outStream << std::endl;

        for (auto iter = m_EventSequence.begin(); iter != m_EventSequence.end();)
        {
            const uint32_t tag = iter->m_Tag;

            // Advance iter until we find the first non-matching tag
            auto tagEndIter = iter;
            for (; tagEndIter != m_EventSequence.end(); ++tagEndIter)
            {
                if (tagEndIter->m_Tag != tag)
                {
                    break;
                }
            }

            outStream << "> Begin Tag: " << tag << std::endl;
            outStream << std::endl;
            AnalyzeEventSequenceAndWriteResults(iter, tagEndIter, outStream);
            outStream << std::endl;
            outStream << "> End Tag: " << tag << std::endl;

            iter = tagEndIter;
        }
    }
}

void Profiler::WaitForDevice(Compute compute) const
{
#if ARMCOMPUTECL_ENABLED
    if(compute == Compute::GpuAcc && g_ProfilingForceGpuSync)
    {
        arm_compute::CLScheduler::get().sync();
    }
#endif
}

const char* Profiler::GetEventComputeDevice(Compute compute) const
{
    switch(compute)
    {
        case Compute::CpuRef:
            return "CpuRef";
        case Compute::CpuAcc:
            return "CpuAcc";
        case Compute::GpuAcc:
            return "GpuAcc";
        default:
            return "Undefined";
    }
}

std::uint32_t Profiler::GetEventColor(Compute compute) const
{
    switch(compute)
    {
        case Compute::CpuRef:
            // Cyan
            return 0xffff001b;
        case Compute::CpuAcc:
            // Green
            return 0x00ff001b;
        case Compute::GpuAcc:
            // Purple
            return 0xff007f1b;
        default:
            // Dark gray
            return 0x5555551b;
    }
}

} // namespace armnn

#endif // ARMNN_PROFILING_ENABLED

