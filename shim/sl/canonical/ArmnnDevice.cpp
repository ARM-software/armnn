//
// Copyright © 2022-2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#define LOG_TAG "arm-armnn-sl"

#include "ArmnnDevice.hpp"

#include <LegacyUtils.h>
#include <OperationsUtils.h>

#include <log/log.h>

#include <memory>
#include <string>

#ifdef __ANDROID__
#include <android/log.h>
#endif

namespace
{

std::string GetBackendString(const armnn_driver::DriverOptions& options)
{
    std::stringstream backends;
    for (auto&& b : options.GetBackends())
    {
        backends << b << " ";
    }
    return backends.str();
}

} // anonymous namespace

namespace armnn_driver
{

using namespace android::nn;

ArmnnDevice::ArmnnDevice(DriverOptions options)
    : m_Runtime(nullptr, nullptr)
    , m_ClTunedParameters(nullptr)
    , m_Options(std::move(options))
{
    // First check if the DriverOptions is happy.
    if (options.ShouldExit())
    {
        // Is this a good or bad exit?
        if (options.GetExitCode() != EXIT_SUCCESS)
        {
            throw armnn::InvalidArgumentException("ArmnnDevice: Insufficient or illegal options specified.");
        }
        else
        {
            throw armnn::InvalidArgumentException("ArmnnDevice: Nothing to do.");
        }
    }

    initVLogMask();
    VLOG(DRIVER) << "ArmnnDevice::ArmnnDevice()";

#ifdef __ANDROID__
    __android_log_print(ANDROID_LOG_DEBUG, "ARMNN_SL", "ArmnnDevice::ArmnnDevice()");
#endif

    armnn::ConfigureLogging(false, m_Options.IsVerboseLoggingEnabled(), armnn::LogSeverity::Trace);
    if (m_Options.IsVerboseLoggingEnabled())
    {
        SetMinimumLogSeverity(android::base::VERBOSE);
    }
    else
    {
        SetMinimumLogSeverity(android::base::INFO);
    }
    armnn::IRuntime::CreationOptions runtimeOptions;

    if (std::find(m_Options.GetBackends().begin(),
                  m_Options.GetBackends().end(),
                  armnn::Compute::GpuAcc) != m_Options.GetBackends().end())
    {
        try
        {
            if (!m_Options.GetClTunedParametersFile().empty())
            {
                m_ClTunedParameters = armnn::IGpuAccTunedParameters::Create(m_Options.GetClTunedParametersMode(),
                                                                            m_Options.GetClTuningLevel());
                try
                {
                    m_ClTunedParameters->Load(m_Options.GetClTunedParametersFile().c_str());
                }
                catch (std::exception& error)
                {
                    // This is only a warning because the file won't exist the first time you are generating it.
                    VLOG(DRIVER) << "ArmnnDevice: Failed to load CL tuned parameters file "
                          << m_Options.GetClTunedParametersFile().c_str() << " : " <<  error.what();
                }
                runtimeOptions.m_GpuAccTunedParameters = m_ClTunedParameters;
            }
        }
        catch (const armnn::ClRuntimeUnavailableException& error)
        {
            VLOG(DRIVER) <<  "ArmnnDevice: Failed to setup CL runtime: %s. Device will be unavailable." << error.what();
        }
        catch (std::exception& error)
        {
            VLOG(DRIVER) <<  "ArmnnDevice: Unknown exception: %s. Device will be unavailable." << error.what();
        }
    }
    runtimeOptions.m_EnableGpuProfiling = m_Options.IsGpuProfilingEnabled();
    m_Runtime = armnn::IRuntime::Create(runtimeOptions);

    std::vector<armnn::BackendId> backends;

    if (m_Runtime)
    {
        const armnn::BackendIdSet supportedDevices = m_Runtime->GetDeviceSpec().GetSupportedBackends();
        for (auto &backend : m_Options.GetBackends())
        {
            if (std::find(supportedDevices.cbegin(), supportedDevices.cend(), backend) == supportedDevices.cend())
            {
                VLOG(DRIVER) << "ArmnnDevice: Requested unknown backend " << backend.Get().c_str();
            }
            else
            {
                if (m_Options.isAsyncModelExecutionEnabled() &&
                    armnn::HasCapability(armnn::BackendOptions::BackendOption{"AsyncExecution", false}, backend))
                {
                    VLOG(DRIVER) << "ArmnnDevice: ArmNN does not support AsyncExecution with the following backend: "
                                 << backend.Get().c_str();
                }
                else
                {
                    backends.push_back(backend);
                }
            }
        }
    }

    if (backends.empty())
    {
        // No known backend specified
        throw armnn::InvalidArgumentException("ArmnnDevice: No known backend specified.");
    }

    m_Options.SetBackends(backends);
    VLOG(DRIVER) << "ArmnnDevice: Created device with the following backends: " << GetBackendString(m_Options).c_str();

#ifdef __ANDROID__
    __android_log_print(ANDROID_LOG_DEBUG,
                        "ARMNN_SL",
                        "ArmnnDevice: Created device with the following backends: %s",
                        GetBackendString(m_Options).c_str());
#endif
}

} // namespace armnn_driver
