//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include "Profiling.hpp"

#include <armnn/BackendId.hpp>
#include <armnn/utility/Assert.hpp>

#include "JsonPrinter.hpp"

#if ARMNN_STREAMLINE_ENABLED
#include <streamline_annotate.h>
#endif

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <map>
#include <stack>

namespace armnn
{

// Controls the amount of memory initially allocated to store profiling events.
// If chosen carefully, the profiling system will not make any additional allocations, thus minimizing its impact on
// measured times.
constexpr std::size_t g_ProfilingEventCountHint = 1024;

// Whether profiling reports should include the sequence of events together with their timings.
constexpr bool g_WriteProfilingEventSequence = true;

// Whether profiling reports should also report detailed information on events grouped by inference.
// This can spam the output stream, so use carefully (or adapt the code to just output information
// of interest).
constexpr bool g_AggregateProfilingEventsByInference = true;

// Whether a call to Profiler::AnalyzeEventsAndWriteResults() will be made when the Profiler is destroyed.
// It can be convenient for local tests.
constexpr bool g_WriteReportToStdOutOnProfilerDestruction = false;

Measurement FindMeasurement(const std::string& name, const Event* event)
{

    ARMNN_ASSERT(event != nullptr);

    // Search though the measurements.
    for (const auto& measurement : event->GetMeasurements())
    {
        if (measurement.m_Name == name)
        {
            // Measurement found.
            return measurement;
        }
    }

    // Measurement not found.
    return Measurement{ "", 0.f, Measurement::Unit::TIME_MS };
}

std::vector<Measurement> FindKernelMeasurements(const Event* event)
{
    ARMNN_ASSERT(event != nullptr);

    std::vector<Measurement> measurements;

    // Search through the measurements.
    for (const auto& measurement : event->GetMeasurements())
    {
        if (measurement.m_Name.rfind("OpenClKernelTimer", 0) == 0
            || measurement.m_Name.rfind("NeonKernelTimer", 0) == 0)
        {
            // Measurement found.
            measurements.push_back(measurement);
        }
    }

    return measurements;
}

std::map<std::string, ProfilerImpl::ProfilingEventStats> ProfilerImpl::CalculateProfilingEventStats() const
{
    std::map<std::string, ProfilingEventStats> nameToStatsMap;

    for (const auto& event : m_EventSequence)
    {
        Measurement measurement = FindMeasurement(WallClockTimer::WALL_CLOCK_TIME, event.get());

        double durationMs = measurement.m_Value;
        auto it = nameToStatsMap.find(event->GetName());
        if (it != nameToStatsMap.end())
        {
            ProfilingEventStats& stats = it->second;
            stats.m_TotalMs += durationMs;
            stats.m_MinMs = std::min(stats.m_MinMs, durationMs);
            stats.m_MaxMs = std::max(stats.m_MaxMs, durationMs);
            ++stats.m_Count;
        }
        else
        {
            nameToStatsMap.emplace(event->GetName(), ProfilingEventStats{ durationMs, durationMs, durationMs, 1 });
        }
    }

    return nameToStatsMap;
}

const Event* GetEventPtr(const Event* ptr) { return ptr;}
const Event* GetEventPtr(const std::unique_ptr<Event>& ptr) {return ptr.get(); }

template<typename ItertType>
void ProfilerImpl::AnalyzeEventSequenceAndWriteResults(ItertType first, ItertType last, std::ostream& outStream) const
{
    // Outputs event sequence, if needed.
    if (g_WriteProfilingEventSequence)
    {
        // Makes sure timestamps are output with 6 decimals, and save old settings.
        std::streamsize oldPrecision = outStream.precision();
        outStream.precision(6);
        std::ios_base::fmtflags oldFlags = outStream.flags();
        outStream.setf(std::ios::fixed);
        // Outputs fields.
        outStream << "Event Sequence - Name | Duration (ms) | Start (ms) | Stop (ms) | Device" << std::endl;
        for (auto event = first; event != last; ++event)
        {
            const Event* eventPtr = GetEventPtr((*event));
            double startTimeMs = FindMeasurement(WallClockTimer::WALL_CLOCK_TIME_START, eventPtr).m_Value;
            double stopTimeMs = FindMeasurement(WallClockTimer::WALL_CLOCK_TIME_STOP, eventPtr).m_Value;

            // Find the WallClock measurement if there is one.
            double durationMs = FindMeasurement(WallClockTimer::WALL_CLOCK_TIME, eventPtr).m_Value;
            outStream << std::setw(50) << eventPtr->GetName() << " "
                      << std::setw(20) << durationMs
                      << std::setw(20) << startTimeMs
                      << std::setw(20) << stopTimeMs
                      << std::setw(20) << eventPtr->GetBackendId().Get()
                      << std::endl;
        }
        outStream << std::endl;
        // Restores previous precision settings.
        outStream.flags(oldFlags);
        outStream.precision(oldPrecision);
    }

    // Aggregates results per event name.
    std::map<std::string, ProfilingEventStats> nameToStatsMap = CalculateProfilingEventStats();

    // Outputs aggregated stats.
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

ProfilerImpl::ProfilerImpl()
    : m_ProfilingEnabled(false),
      m_DetailsToStdOutMethod(ProfilingDetailsMethod::Undefined)
{
    m_EventSequence.reserve(g_ProfilingEventCountHint);

#if ARMNN_STREAMLINE_ENABLED
    // Initialises streamline annotations.
    ANNOTATE_SETUP;
#endif
}

ProfilerImpl::~ProfilerImpl()
{
    if (m_ProfilingEnabled)
    {
        if (g_WriteReportToStdOutOnProfilerDestruction)
        {
            Print(std::cout);
        }
    }

    // Un-register this profiler from the current thread.
    ProfilerManager::GetInstance().RegisterProfiler(nullptr);
}

bool ProfilerImpl::IsProfilingEnabled()
{
    return m_ProfilingEnabled;
}

void ProfilerImpl::EnableProfiling(bool enableProfiling)
{
    m_ProfilingEnabled = enableProfiling;
}

void ProfilerImpl::EnableNetworkDetailsToStdOut(ProfilingDetailsMethod details)
{
    m_DetailsToStdOutMethod = details;
}

Event* ProfilerImpl::BeginEvent(armnn::IProfiler* profiler,
                                const BackendId& backendId,
                                const std::string& label,
                                std::vector<InstrumentPtr>&& instruments,
                                const Optional<profiling::ProfilingGuid>& guid)
{
    Event* parent = m_Parents.empty() ? nullptr : m_Parents.top();
    m_EventSequence.push_back(std::make_unique<Event>(label,
                                                      profiler,
                                                      parent,
                                                      backendId,
                                                      std::move(instruments),
                                                      guid));
    Event* event = m_EventSequence.back().get();
    event->Start();

#if ARMNN_STREAMLINE_ENABLED
    ANNOTATE_CHANNEL_COLOR(uint32_t(m_Parents.size()), GetEventColor(backendId), label.c_str());
#endif

    m_Parents.push(event);
    return event;
}

void ProfilerImpl::EndEvent(Event* event)
{
    event->Stop();

    ARMNN_ASSERT(!m_Parents.empty());
    ARMNN_ASSERT(event == m_Parents.top());
    m_Parents.pop();

    Event* parent = m_Parents.empty() ? nullptr : m_Parents.top();
    IgnoreUnused(parent);
    ARMNN_ASSERT(event->GetParentEvent() == parent);

#if ARMNN_STREAMLINE_ENABLED
    ANNOTATE_CHANNEL_END(uint32_t(m_Parents.size()));
#endif
}

int CalcLevel(const Event* eventPtr)
{
    int level = 0;
    while (eventPtr != nullptr)
    {
        eventPtr = eventPtr->GetParentEvent();
        level++;
    }
    return level;
}

void ProfilerImpl::PopulateParent(std::vector<const Event*>& outEvents, int& outBaseLevel, std::string parentName) const
{
    outEvents.reserve(m_EventSequence.size());
    for (const auto& event : m_EventSequence)
    {
        const Event* eventPtrRaw = event.get();
        if (eventPtrRaw->GetName() == parentName)
        {
            outBaseLevel = (outBaseLevel == -1) ? CalcLevel(eventPtrRaw) : outBaseLevel;
            outEvents.push_back(eventPtrRaw);
        }
    }
}

void ProfilerImpl::PopulateDescendants(std::map<const Event*, std::vector<const Event*>>& outDescendantsMap) const
{
    for (const auto& event : m_EventSequence)
    {
        const Event* eventPtrRaw = event.get();
        const Event* parent = eventPtrRaw->GetParentEvent();

        if (!parent)
        {
            continue;
        }

        auto it = outDescendantsMap.find(parent);
        if (it == outDescendantsMap.end())
        {
            outDescendantsMap.emplace(parent, std::vector<const Event*>({ eventPtrRaw }));
        }
        else
        {
            it->second.push_back(eventPtrRaw);
        }
    }
}

void ConfigureDetailsObject(JsonChildObject& detailsObject,
                            std::string layerDetailsStr)
{
    detailsObject.SetType(JsonObjectType::ExecObjectDesc);
    detailsObject.SetAndParseDetails(layerDetailsStr);

}

void ExtractJsonObjects(unsigned int inferenceIndex,
                        const Event* parentEvent,
                        JsonChildObject& parentObject,
                        std::map<const Event*, std::vector<const Event*>> descendantsMap)
{
    ARMNN_ASSERT(parentEvent);

    // If profiling GUID is entered, process it
    if (parentEvent->GetProfilingGuid().has_value())
    {
        profiling::ProfilingGuid profilingGuid;
        profilingGuid = parentEvent->GetProfilingGuid().value();
        parentObject.SetGuid(profilingGuid);
    }
    std::vector<Measurement> instrumentMeasurements = parentEvent->GetMeasurements();
    unsigned int childIdx = 0;
    for (size_t measurementIndex = 0; measurementIndex < instrumentMeasurements.size(); ++measurementIndex, ++childIdx)
    {
        if (inferenceIndex == 0)
        {
            // Only add kernel measurement once, in case of multiple inferences
            JsonChildObject measurementObject{ instrumentMeasurements[measurementIndex].m_Name };
            measurementObject.SetUnit(instrumentMeasurements[measurementIndex].m_Unit);
            measurementObject.SetType(JsonObjectType::Measurement);

            ARMNN_ASSERT(parentObject.NumChildren() == childIdx);
            parentObject.AddChild(measurementObject);
        }

        parentObject.GetChild(childIdx).AddMeasurement(instrumentMeasurements[measurementIndex].m_Value);
    }

    auto childEventsIt = descendantsMap.find(parentEvent);
    if (childEventsIt != descendantsMap.end())
    {
        for (auto childEvent : childEventsIt->second)
        {
            if (inferenceIndex == 0)
            {
                // Only add second level once, in case of multiple inferences
                JsonChildObject childObject{ childEvent->GetName() };
                childObject.SetType(JsonObjectType::Event);
                parentObject.AddChild(childObject);
            }

            // Recursively process children. In reality this won't be very deep recursion. ~4-6 levels deep.
            ExtractJsonObjects(inferenceIndex, childEvent, parentObject.GetChild(childIdx), descendantsMap);

            childIdx++;
        }
    }
}

void ProfilerImpl::Print(std::ostream& outStream) const
{
    // Makes sure timestamps are output with 6 decimals, and save old settings.
    std::streamsize oldPrecision = outStream.precision();
    outStream.precision(6);
    std::ios_base::fmtflags oldFlags = outStream.flags();
    outStream.setf(std::ios::fixed);
    JsonPrinter printer(outStream);

    // First find all the parent Events and print out duration measurements.
    int baseLevel = -1;

    std::vector<const Event*> optimizations;
    PopulateParent(optimizations, baseLevel, "Optimizer");

    std::vector<const Event*> loadedNetworks;
    PopulateParent(loadedNetworks, baseLevel, "LoadedNetwork");

    std::vector<const Event*> inferences;
    PopulateParent(inferences, baseLevel, "EnqueueWorkload");

    // Second map out descendants hierarchy
    std::map<const Event*, std::vector<const Event*>> descendantsMap;
    PopulateDescendants(descendantsMap);

    // Extract json objects for each parent event type
    JsonChildObject optimizeObject{ "optimize_measurements" };

    for (unsigned int optimizeIndex = 0; optimizeIndex < optimizations.size(); ++optimizeIndex)
    {
        auto optimization = optimizations[optimizeIndex];
        ExtractJsonObjects(optimizeIndex, optimization, optimizeObject, descendantsMap);
    }

    JsonChildObject loadedNetworkObject{ "loaded_network_measurements" };

    for (unsigned int loadedNetworkIndex = 0; loadedNetworkIndex < loadedNetworks.size(); ++loadedNetworkIndex)
    {
        auto loadedNetwork = loadedNetworks[loadedNetworkIndex];
        ExtractJsonObjects(loadedNetworkIndex, loadedNetwork, loadedNetworkObject, descendantsMap);
    }

    JsonChildObject inferenceObject{ "inference_measurements" };

    for (unsigned int inferenceIndex = 0; inferenceIndex < inferences.size(); ++inferenceIndex)
    {
        auto inference = inferences[inferenceIndex];
        ExtractJsonObjects(inferenceIndex, inference, inferenceObject, descendantsMap);
    }

    printer.PrintHeader();
    printer.PrintArmNNHeader();

    if (m_ProfilingDetails.get()->DetailsExist() &&
        (m_DetailsToStdOutMethod == ProfilingDetailsMethod::DetailsOnly
         || m_DetailsToStdOutMethod == ProfilingDetailsMethod::DetailsWithEvents))
    {
        JsonChildObject detailsObject{ "layer_details" };
        if (m_DetailsToStdOutMethod == ProfilingDetailsMethod::DetailsOnly)
        {
            detailsObject.EnableDetailsOnly();
        }
        detailsObject.SetType(JsonObjectType::ExecObjectDesc);
        detailsObject.SetAndParseDetails(m_ProfilingDetails.get()->GetProfilingDetails());

        size_t id = 0;
        printer.PrintJsonChildObject(detailsObject, id);
    }

    // print inference object, also prints child layer and kernel measurements
    size_t id = 0;
    if (m_DetailsToStdOutMethod != ProfilingDetailsMethod::DetailsOnly)
    {
        printer.PrintJsonChildObject(optimizeObject, id);
        printer.PrintSeparator();
        printer.PrintNewLine();
        printer.PrintJsonChildObject(loadedNetworkObject, id);
        printer.PrintSeparator();
        printer.PrintNewLine();
        printer.PrintJsonChildObject(inferenceObject, id);
        printer.PrintNewLine();
    }
    // end of ArmNN
    printer.PrintFooter();

    // end of main JSON object
    printer.PrintNewLine();
    printer.PrintFooter();
    printer.PrintNewLine();

    // Restores previous precision settings.
    outStream.flags(oldFlags);
    outStream.precision(oldPrecision);

    if (m_DetailsToStdOutMethod == ProfilingDetailsMethod::DetailsOnly)
    {
        exit(0);
    }
}

void ProfilerImpl::AnalyzeEventsAndWriteResults(std::ostream& outStream) const
{
    // Stack should be empty now.
    const bool saneMarkerSequence = m_Parents.empty();

    // Abort if the sequence of markers was found to have incorrect information:
    // The stats cannot be trusted.
    if (!saneMarkerSequence)
    {
        outStream << "Cannot write profiling stats. "
                     "Unexpected errors were found when analyzing the sequence of logged events, "
                     "which may lead to plainly wrong stats. The profiling system may contain implementation "
                     "issues or could have been used in an unsafe manner." << std::endl;
        return;
    }

    // Analyzes the full sequence of events.
    AnalyzeEventSequenceAndWriteResults(m_EventSequence.cbegin(),
                                        m_EventSequence.cend(),
                                        outStream);

    // Aggregates events by tag if requested (spams the output stream if done for all tags).
    if (g_AggregateProfilingEventsByInference)
    {
        outStream << std::endl;
        outStream << "***" << std::endl;
        outStream << "*** Per Inference Stats" << std::endl;
        outStream << "***" << std::endl;
        outStream << std::endl;

        int baseLevel = -1;
        std::vector<const Event*> inferences;
        PopulateParent(inferences, baseLevel, "EnqueueWorkload");

        // Second map out descendants hierarchy
        std::map<const Event*, std::vector<const Event*>> descendantsMap;
        PopulateDescendants(descendantsMap);

        std::function<void(const Event*, std::vector<const Event*>&)>
            FindDescendantEvents = [&](const Event* eventPtr, std::vector<const Event*>& sequence)
            {
                sequence.push_back(eventPtr);

                if (CalcLevel(eventPtr) > baseLevel+2) //We only care about levels as deep as workload executions.
                {
                    return;
                }

                auto children = descendantsMap.find(eventPtr);
                if (children == descendantsMap.end())
                {
                    return;
                }

                if (!(children->second.empty()))
                {
                    return FindDescendantEvents(children->second[0], sequence);
                }
            };

        // Third, find events belonging to each inference
        int inferenceIdx = 0;
        for (auto inference : inferences)
        {
            std::vector<const Event*> sequence;

            //build sequence, depth first
            FindDescendantEvents(inference, sequence);

            outStream << "> Begin Inference: " << inferenceIdx << std::endl;
            outStream << std::endl;
            AnalyzeEventSequenceAndWriteResults(sequence.cbegin(),
                                                sequence.cend(),
                                                outStream);
            outStream << std::endl;
            outStream << "> End Inference: " << inferenceIdx << std::endl;

            inferenceIdx++;
        }
    }
}

std::uint32_t ProfilerImpl::GetEventColor(const BackendId& backendId) const
{
    static BackendId cpuRef("CpuRef");
    static BackendId cpuAcc("CpuAcc");
    static BackendId gpuAcc("GpuAcc");
    if (backendId == cpuRef)
    {
        // Cyan
        return 0xffff001b;
    }
    else if (backendId == cpuAcc)
    {
        // Green
        return 0x00ff001b;
    }
    else if (backendId == gpuAcc)
    {
        // Purple
        return 0xff007f1b;
    }
    else
    {
        // Dark gray
        return 0x5555551b;
    }
}

// The thread_local pointer to the profiler instance.
thread_local IProfiler* tl_Profiler = nullptr;

ProfilerManager& ProfilerManager::GetInstance()
{
    // Global reference to the single ProfileManager instance allowed.
    static ProfilerManager s_ProfilerManager;
    return s_ProfilerManager;
}

void ProfilerManager::RegisterProfiler(IProfiler* profiler)
{
    tl_Profiler = profiler;
}

IProfiler* ProfilerManager::GetProfiler()
{
    return tl_Profiler;
}

void IProfiler::EnableProfiling(bool enableProfiling)
{
    pProfilerImpl->EnableProfiling(enableProfiling);
}

void IProfiler::EnableNetworkDetailsToStdOut(ProfilingDetailsMethod detailsMethod)
{
    pProfilerImpl->EnableNetworkDetailsToStdOut(detailsMethod);
}

bool IProfiler::IsProfilingEnabled()
{
    return pProfilerImpl->IsProfilingEnabled();
}

void IProfiler::AnalyzeEventsAndWriteResults(std::ostream& outStream) const
{
    pProfilerImpl->AnalyzeEventsAndWriteResults(outStream);
}

void IProfiler::Print(std::ostream& outStream) const
{
    pProfilerImpl->Print(outStream);
}

Event* IProfiler::BeginEvent(const BackendId& backendId,
                             const std::string& label,
                             std::vector<InstrumentPtr>&& instruments,
                             const Optional<profiling::ProfilingGuid>& guid)
{
    return pProfilerImpl->BeginEvent(this, backendId, label, std::move(instruments), guid);
}

IProfiler::~IProfiler() = default;
IProfiler::IProfiler() : pProfilerImpl(new ProfilerImpl())
{};

} // namespace armnn
