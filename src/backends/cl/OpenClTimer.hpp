//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <Instrument.hpp>

#include <arm_compute/runtime/CL/CLScheduler.h>
#include <arm_compute/core/CL/OpenCL.h>

#include <vector>
#include <list>

namespace armnn
{

/// OpenClTimer instrument that times all OpenCl kernels executed between calls to Start() and Stop().
class OpenClTimer : public Instrument
{
public:
    OpenClTimer();
    ~OpenClTimer() = default;

    /// Start the OpenCl timer
    void Start() override;

    /// Stop the OpenCl timer
    void Stop() override;

    /// Return true if this Instrument has kernels for recording measurements
    bool HasKernelMeasurements() const override;

    /// Get the name of the timer
    /// \return Name of the timer
    const char* GetName() const override { return "OpenClKernelTimer"; }

    /// Get the recorded measurements. This will be a list of the execution durations for all the OpenCl kernels.
    /// \return Recorded measurements
    std::vector<Measurement> GetMeasurements() const override;

private:
    using CLScheduler = arm_compute::CLScheduler;
    using CLSymbols = arm_compute::CLSymbols;
    using ClEvent = cl::Event;
    using ClEnqueueFunc = decltype(CLSymbols::clEnqueueNDRangeKernel_ptr);

    /// Stores info about the OpenCl kernel
    struct KernelInfo
    {
        KernelInfo(const std::string& name, cl_event& event) : m_Name(name), m_Event(event) {}

        std::string m_Name;
        ClEvent m_Event;
    };

    std::list<KernelInfo>                m_Kernels; ///< List of all kernels executed
    ClEnqueueFunc                        m_OriginalEnqueueFunction; ///< Keep track of original OpenCl function
};

} //namespace armnn