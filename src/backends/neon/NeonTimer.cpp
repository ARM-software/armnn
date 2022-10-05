//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "NeonTimer.hpp"
#include "NeonInterceptorScheduler.hpp"

#include <armnn/utility/Assert.hpp>
#include <armnn/utility/PolymorphicDowncast.hpp>

#include <memory>

namespace armnn
{
namespace
{
static thread_local auto g_Interceptor = std::make_shared<NeonInterceptorScheduler>(arm_compute::Scheduler::get());
}

void NeonTimer::Start()
{
    m_Kernels.clear();
    ARMNN_ASSERT(g_Interceptor->GetKernels() == nullptr);
    g_Interceptor->SetKernels(&m_Kernels);

    m_RealSchedulerType = arm_compute::Scheduler::get_type();
    //Note: We can't currently replace a custom scheduler
    if(m_RealSchedulerType != arm_compute::Scheduler::Type::CUSTOM)
    {
        // Keep the real schedule and add NeonInterceptorScheduler as an interceptor
        m_RealScheduler  = &arm_compute::Scheduler::get();
        arm_compute::Scheduler::set(armnn::PolymorphicPointerDowncast<arm_compute::IScheduler>(g_Interceptor));
    }
}

void NeonTimer::Stop()
{
    // Restore real scheduler
    g_Interceptor->SetKernels(nullptr);
    arm_compute::Scheduler::set(m_RealSchedulerType);
    m_RealScheduler = nullptr;
}

bool NeonTimer::HasKernelMeasurements() const
{
    return m_Kernels.size() > 0;
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
