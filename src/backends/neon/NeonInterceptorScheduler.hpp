//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include "NeonTimer.hpp"
#include "WallClockTimer.hpp"

#include <arm_compute/runtime/IScheduler.h>
#include <arm_compute/runtime/Scheduler.h>
#include <arm_compute/core/CPP/ICPPKernel.h>

namespace armnn
{

class NeonInterceptorScheduler : public arm_compute::IScheduler
{
public:
    NeonInterceptorScheduler(arm_compute::IScheduler &realScheduler);
    ~NeonInterceptorScheduler() = default;

    void set_num_threads(unsigned int numThreads) override;

    unsigned int num_threads() const override;

    void schedule(arm_compute::ICPPKernel *kernel, const Hints &hints) override;

    void run_workloads(std::vector<Workload> &workloads) override;

    void run_tagged_workloads(std::vector<Workload> &workloads, const char *tag) override;

    void SetKernels(NeonTimer::KernelMeasurements* kernels) { m_Kernels = kernels; }
    NeonTimer::KernelMeasurements* GetKernels() { return m_Kernels; }

    void schedule_op(arm_compute::ICPPKernel* kernel,
                     const Hints& hints,
                     const arm_compute::Window& window,
                     arm_compute::ITensorPack& tensors ) override;
private:
    NeonTimer::KernelMeasurements* m_Kernels;
    arm_compute::IScheduler& m_RealScheduler;
};

} // namespace armnn
