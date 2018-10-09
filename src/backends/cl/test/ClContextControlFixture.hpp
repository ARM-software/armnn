//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <backends/cl/ClContextControl.hpp>

template<bool ProfilingEnabled>
struct ClContextControlFixtureBase
{
    static ClContextControlFixtureBase*& Instance()
    {
        static ClContextControlFixtureBase* s_Instance = nullptr;
        return s_Instance;
    }

    // Initialising ClContextControl to ensure OpenCL is loaded correctly for each test case
    ClContextControlFixtureBase()
        : m_ClContextControl(nullptr, ProfilingEnabled)
    {
        Instance() = this;
    }
    ~ClContextControlFixtureBase()
    {
        Instance() = nullptr;
    }

    armnn::ClContextControl m_ClContextControl;
};

using ClContextControlFixture = ClContextControlFixtureBase<false>;
using ClProfilingContextControlFixture = ClContextControlFixtureBase<true>;
