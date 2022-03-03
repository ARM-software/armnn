//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once
#include <chrono>
#include <iostream>
#include <string>

using namespace std::chrono;

namespace common
{
/**
* @brief Used for meausuring performance of specific actions in the code.
 * Profiling should be enabled with a parameter passed to the constructor and
 * it's disabled by default.
 * In order to measure timing, wrap the desired code section with
 * ProfilingStart() and ProfilingStopAndPrintUs(title)
*/
class Profiling {
private:

    struct group_thousands : std::numpunct<char>
    {
        std::string do_grouping() const override { return "\3"; }
    };

    bool mProfilingEnabled{};
    steady_clock::time_point mStart{};
    steady_clock::time_point mStop{};
public:
    Profiling() : mProfilingEnabled(false) {};

    /**
    * @brief Initializes the profiling object.
    *
    *       * @param[in] isEnabled - Enables the profiling computation and prints.
    */
    explicit Profiling(bool isEnabled) : mProfilingEnabled(isEnabled) {};

/**
* @brief Starts the profiling measurement.
*
*/

    void ProfilingStart()
    {
        if (mProfilingEnabled)
        {
            mStart = steady_clock::now();
        }
    }

/**
* @brief Stops the profiling measurement, without printing the results.
*
*/
    auto ProfilingStop()
    {
        if (mProfilingEnabled)
        {
            mStop = steady_clock::now();
        }
    }

/**
* @brief Get the measurement result in micro-seconds.
*
*/
    auto ProfilingGetUs()
    {
        return mProfilingEnabled ? duration_cast<microseconds>(mStop - mStart).count() : 0;
    }

/**
* @brief Stop the profiling measurement and print the result in micro-seconds.
*
*/
    void ProfilingStopAndPrintUs(const std::string &title)
    {
        ProfilingStop();
        if (mProfilingEnabled) {
            std::cout.imbue(std::locale(std::cout.getloc(), new group_thousands));
            std::cout << "Profiling: " << title << ": " << ProfilingGetUs() << " uSeconds" << std::endl;
        }
    }
};
}// namespace common