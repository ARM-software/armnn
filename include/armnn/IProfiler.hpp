//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <iostream>

namespace armnn
{

class IProfiler
{
public:
    /// Enables/disables profiling for this profiler.
    /// @param [in] enableProfiling A flag that indicates whether profiling should be enabled or not.
    virtual void EnableProfiling(bool enableProfiling) = 0;

    /// Checks whether profiling is enabled.
    /// Profiling is disabled by default.
    /// @return true if profiling is enabled, false otherwise.
    virtual bool IsProfilingEnabled() = 0;

    /// Analyzes the tracked events and writes the results to the given output stream.
    /// Please refer to the configuration variables in Profiling.cpp to customize the information written.
    /// @param [out] outStream The stream where to write the profiling results to.
    virtual void AnalyzeEventsAndWriteResults(std::ostream& outStream) const = 0;

    /// Print stats for events in JSON Format to the given output stream.
    /// @param [out] outStream The stream where to write the profiling results to.
    virtual void Print(std::ostream& outStream) const = 0;

protected:
    ~IProfiler() {}
};

} // namespace armnn
