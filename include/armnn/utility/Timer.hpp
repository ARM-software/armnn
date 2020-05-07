//
// Copyright © 2020 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <chrono>
#include <iomanip>

namespace armnn
{

inline std::chrono::high_resolution_clock::time_point GetTimeNow()
{
    return std::chrono::high_resolution_clock::now();
}

inline std::chrono::duration<double, std::milli> GetTimeDuration(
        std::chrono::high_resolution_clock::time_point start_time)
{
    return std::chrono::duration<double, std::milli>(GetTimeNow() - start_time);
}

}