//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "Instrument.hpp"

#include <arm_compute/runtime/IScheduler.h>
#include <arm_compute/runtime/Scheduler.h>
#include <arm_compute/core/CPP/ICPPKernel.h>

#include <chrono>
#include <map>
#include <list>

namespace armnn
{

class NeonTimer : public Instrument
{
public:
    using KernelMeasurements = std::vector<Measurement>;

    NeonTimer() = default;
    ~NeonTimer() = default;

    void Start() override;

    void Stop() override;

    bool HasKernelMeasurements() const override;

    std::vector<Measurement> GetMeasurements() const override;

    const char* GetName() const override;

private:
    KernelMeasurements m_Kernels;
    arm_compute::IScheduler* m_RealScheduler;
    arm_compute::Scheduler::Type m_RealSchedulerType;
};

}