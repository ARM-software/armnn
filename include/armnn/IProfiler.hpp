//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <iostream>
#include <memory>
#include <vector>

namespace armnn
{

class ProfilerImpl;
class BackendId;
class Instrument;
class Event;
class IProfiler
{
public:
    /// Enables/disables profiling for this profiler.
    /// @param [in] enableProfiling A flag that indicates whether profiling should be enabled or not.
    void EnableProfiling(bool enableProfiling);

    /// Checks whether profiling is enabled.
    /// Profiling is disabled by default.
    /// @return true if profiling is enabled, false otherwise.
    bool IsProfilingEnabled();

    /// Analyzes the tracked events and writes the results to the given output stream.
    /// Please refer to the configuration variables in Profiling.cpp to customize the information written.
    /// @param [out] outStream The stream where to write the profiling results to.
    void AnalyzeEventsAndWriteResults(std::ostream& outStream) const;

    /// Print stats for events in JSON Format to the given output stream.
    /// @param [out] outStream The stream where to write the profiling results to.
    void Print(std::ostream& outStream) const;

    ~IProfiler();
    IProfiler();

private:
    using InstrumentPtr = std::unique_ptr<Instrument>;
    Event* BeginEvent(const BackendId& backendId,
                      const std::string& label,
                      std::vector<InstrumentPtr>&& instruments);
    std::unique_ptr<ProfilerImpl> pProfilerImpl;
    friend class ScopedProfilingEvent;

    // Friend functions for unit testing, see ProfilerTests.cpp.
    friend size_t GetProfilerEventSequenceSize(armnn::IProfiler* profiler);
};

} // namespace armnn
