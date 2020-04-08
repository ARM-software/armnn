//
// Copyright © 2020 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
%{
#include "armnn/IProfiler.hpp"
%}

namespace armnn
{

%feature("docstring",
"
Interface for profiling Arm NN. See `IRuntime.GetProfiler`.

IProfiler object allows you to enable profiling and get various profiling results.

") IProfiler;
%nodefaultctor IProfiler;
%nodefaultdtor IProfiler;
class IProfiler
{
public:

   %feature("docstring",
    "
    Sets the profiler to start/stop profiling.

    Args:
        enableProfiling (bool): Flag to enable/disable profiling.

    ") EnableProfiling;

  void EnableProfiling(bool enableProfiling);

  %feature("docstring",
    "
    Checks if profiling is enabled.

    Returns:
        bool: If profiling is enabled or not.

    ") IsProfilingEnabled;

  bool IsProfilingEnabled();
};

%extend IProfiler {

   %feature("docstring",
   "
   Gets the string value of the profiling events analysis log.

   Returns:
       str: The profiling events analysis log.

   ") event_log;

   std::string event_log()
   {
       std::ostringstream oss;
       $self->AnalyzeEventsAndWriteResults(oss);
       return oss.str();
   }

   %feature("docstring",
   "
   Gets the profiling log as the JSON string.

   Returns:
       str: Profiling log as JSON formatted string.

   ") as_json;

   std::string as_json()
   {
       std::ostringstream oss;
       $self->Print(oss);
       return oss.str();
   }
}
}
