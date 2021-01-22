//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "NeonInterceptorScheduler.hpp"

namespace armnn{

NeonInterceptorScheduler::NeonInterceptorScheduler(arm_compute::IScheduler &realScheduler)
        : m_Kernels(nullptr), m_RealScheduler(realScheduler)
{
}

void NeonInterceptorScheduler::set_num_threads(unsigned int numThreads)
{
    m_RealScheduler.set_num_threads(numThreads);
}

unsigned int NeonInterceptorScheduler::num_threads() const
{
    return m_RealScheduler.num_threads();
}

void NeonInterceptorScheduler::schedule(arm_compute::ICPPKernel* kernel, const Hints& hints)
{
    WallClockTimer::clock::time_point startTime = WallClockTimer::clock::now();
    m_RealScheduler.schedule(kernel, hints.split_dimension());
    WallClockTimer::clock::time_point stopTime = WallClockTimer::clock::now();

    const auto delta       = std::chrono::duration<double, std::micro>(stopTime - startTime);
    m_Kernels->emplace_back(kernel->name(), delta.count(), Measurement::Unit::TIME_US);
}

void NeonInterceptorScheduler::run_workloads(std::vector <Workload>& workloads)
{
    WallClockTimer::clock::time_point startTime = WallClockTimer::clock::now();
    m_RealScheduler.run_tagged_workloads(workloads, nullptr);
    WallClockTimer::clock::time_point stopTime = WallClockTimer::clock::now();

    const auto delta       = std::chrono::duration<double, std::micro>(stopTime - startTime);
    m_Kernels->emplace_back(std::string("Workload"), delta.count(), Measurement::Unit::TIME_US);
}

void NeonInterceptorScheduler::run_tagged_workloads(std::vector<Workload> &workloads, const char *tag)
{
    WallClockTimer::clock::time_point startTime = WallClockTimer::clock::now();
    m_RealScheduler.run_tagged_workloads(workloads, tag);
    WallClockTimer::clock::time_point stopTime = WallClockTimer::clock::now();

    const auto delta       = std::chrono::duration<double, std::micro>(stopTime - startTime);
    m_Kernels->emplace_back(std::string(tag != nullptr ? tag : "Unknown"), delta.count(), Measurement::Unit::TIME_US);
}

void NeonInterceptorScheduler::schedule_op(arm_compute::ICPPKernel* kernel,
                                           const Hints& hints,
                                           const arm_compute::Window& window,
                                           arm_compute::ITensorPack& tensors )
{

    WallClockTimer::clock::time_point startTime = WallClockTimer::clock::now();
    m_RealScheduler.schedule_op(kernel, hints, window, tensors);
    WallClockTimer::clock::time_point stopTime = WallClockTimer::clock::now();

    const auto delta       = std::chrono::duration<double, std::micro>(stopTime - startTime);
    m_Kernels->emplace_back(kernel->name(), delta.count(), Measurement::Unit::TIME_US);
}

} // namespace armnn