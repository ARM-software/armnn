//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <backends/cl/ClContextControl.hpp>

template<bool ProfilingEnabled>
struct ClContextControlFixtureBase
{
    // Initialising ClContextControl to ensure OpenCL is loaded correctly for each test case
    ClContextControlFixtureBase() : m_ClContextControl(nullptr, ProfilingEnabled) {}
    ~ClContextControlFixtureBase() {}

    armnn::ClContextControl m_ClContextControl;
};

using ClContextControlFixture = ClContextControlFixtureBase<false>;
using ClProfilingContextControlFixture = ClContextControlFixtureBase<true>;
