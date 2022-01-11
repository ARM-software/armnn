//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "OpenClTimer.hpp"

#include <armnn/utility/IgnoreUnused.hpp>

#include <string>
#include <sstream>


namespace armnn
{

OpenClTimer::OpenClTimer()
{
}

void OpenClTimer::Start()
{
    m_Kernels.clear();

    auto interceptor = [this](  cl_command_queue command_queue,
                                cl_kernel        kernel,
                                cl_uint          work_dim,
                                const size_t    *gwo,
                                const size_t    *gws,
                                const size_t    *lws,
                                cl_uint          num_events_in_wait_list,
                                const cl_event * event_wait_list,
                                cl_event *       event)
        {
            IgnoreUnused(event);
            cl_int retVal = 0;

            // Get the name of the kernel
            cl::Kernel retainedKernel(kernel, true);
            std::stringstream ss;
            ss << retainedKernel.getInfo<CL_KERNEL_FUNCTION_NAME>();

            // Embed workgroup sizes into the name
            if(gws != nullptr)
            {
                ss << " GWS[" << gws[0] << "," << gws[1] << "," << gws[2] << "]";
            }
            if(lws != nullptr)
            {
                ss << " LWS[" << lws[0] << "," << lws[1] << "," << lws[2] << "]";
            }

            cl_event customEvent;

            // Forward to original OpenCl function
            retVal = m_OriginalEnqueueFunction( command_queue,
                                                kernel,
                                                work_dim,
                                                gwo,
                                                gws,
                                                lws,
                                                num_events_in_wait_list,
                                                event_wait_list,
                                                &customEvent);

            // Store the Kernel info for later GetMeasurements() call
            m_Kernels.emplace_back(ss.str(), customEvent);

            if(event != nullptr)
            {
                //return cl_event from the intercepted call
                clRetainEvent(customEvent);
                *event = customEvent;
            }

            return retVal;
        };

    m_OriginalEnqueueFunction = CLSymbols::get().clEnqueueNDRangeKernel_ptr;
    CLSymbols::get().clEnqueueNDRangeKernel_ptr = interceptor;
}

void OpenClTimer::Stop()
{
    CLSymbols::get().clEnqueueNDRangeKernel_ptr = m_OriginalEnqueueFunction;
}

std::vector<Measurement> OpenClTimer::GetMeasurements() const
{
    std::vector<Measurement> measurements;

    cl_command_queue_properties clQueueProperties = CLScheduler::get().queue().getInfo<CL_QUEUE_PROPERTIES>();

    int idx = 0;
    for (auto& kernel : m_Kernels)
    {
        std::string name = std::string(this->GetName()) + "/" + std::to_string(idx++) + ": " + kernel.m_Name;

        double timeUs = 0.0;
        if((clQueueProperties & CL_QUEUE_PROFILING_ENABLE) != 0)
        {
            // Wait for the event to finish before accessing profile results.
            kernel.m_Event.wait();

            cl_ulong start = kernel.m_Event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
            cl_ulong end   = kernel.m_Event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
            timeUs = static_cast<double>(end - start) / 1000.0;
        }

        measurements.emplace_back(name, timeUs, Measurement::Unit::TIME_US);
    }

    return measurements;
}

} //namespace armnn
