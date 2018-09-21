//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "NeonInterceptorScheduler.hpp"

#include <boost/assert.hpp>

namespace armnn{

NeonInterceptorScheduler::NeonInterceptorScheduler(NeonTimer::KernelMeasurements& kernels,
                                                   arm_compute::IScheduler &realScheduler)
        : m_Kernels(kernels), m_RealScheduler(realScheduler)
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
    m_Timer.Start();
    m_RealScheduler.schedule(kernel, hints.split_dimension());
    m_Timer.Stop();

    m_Timer.SetScaleFactor(Measurement::Unit::TIME_US);
    std::vector<Measurement> measurements = m_Timer.GetMeasurements();
    BOOST_ASSERT(!measurements.empty());

    Measurement measurement(measurements.front()); // NOTE: 1st measurement is delta
    measurement.m_Name = kernel->name();
    m_Kernels.push_back(std::move(measurement));
}

void NeonInterceptorScheduler::run_workloads(std::vector <Workload>& workloads)
{
    m_Timer.Start();
    // NOTE: we should think about utilising the tag to make profiling more understandable
    m_RealScheduler.run_tagged_workloads(workloads, nullptr);
    m_Timer.Stop();

    m_Timer.SetScaleFactor(Measurement::Unit::TIME_US);
    std::vector<Measurement> measurements = m_Timer.GetMeasurements();
    BOOST_ASSERT_MSG(measurements.size() == 3, "WallClockTimer does not have correct amount of measurements.");

    // WallClockTimer has 3 measurements, duration always being the first.
    Measurement measurement(measurements.front());
    measurement.m_Name = "Workload";
    m_Kernels.push_back(std::move(measurement));
}

} // namespace armnn