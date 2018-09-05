//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "NeonTimer.hpp"
#include "NeonInterceptorScheduler.hpp"

#include <memory>

#include <boost/assert.hpp>
#include <boost/format.hpp>

namespace armnn
{

void NeonTimer::Start()
{
    m_Kernels.clear();
    m_RealSchedulerType = arm_compute::Scheduler::get_type();
    //Note: We can't currently replace a custom scheduler
    if(m_RealSchedulerType != arm_compute::Scheduler::Type::CUSTOM)
    {
        // Keep the real schedule and add NeonInterceptorScheduler as an interceptor
        m_RealScheduler  = &arm_compute::Scheduler::get();
        auto interceptor = std::make_shared<NeonInterceptorScheduler>(m_Kernels, *m_RealScheduler);
        arm_compute::Scheduler::set(std::static_pointer_cast<arm_compute::IScheduler>(interceptor));
    }
}

void NeonTimer::Stop()
{
    // Restore real scheduler
    arm_compute::Scheduler::set(m_RealSchedulerType);
    m_RealScheduler = nullptr;
}

std::vector<Measurement> NeonTimer::GetMeasurements() const
{
    std::vector<Measurement> measurements = m_Kernels;
    unsigned int kernel_number = 0;
    for (auto & kernel : measurements)
    {
        std::string kernelName = std::string(this->GetName()) + "/" + std::to_string(kernel_number++) + ": " + kernel
                .m_Name;
        kernel.m_Name = kernelName;
    }
    return measurements;
}

const char* NeonTimer::GetName() const
{
    return "NeonKernelTimer";
}

}
