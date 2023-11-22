//
// Copyright © 2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "GpuFsaBackendContext.hpp"
#include "GpuFsaContextControl.hpp"

#include <armnn/utility/Assert.hpp>
#include <armnn/utility/PolymorphicDowncast.hpp>

#include <arm_compute/core/CL/OpenCL.h>
#include <arm_compute/core/CL/CLKernelLibrary.h>
#include <arm_compute/runtime/CL/CLScheduler.h>
#include <arm_compute/runtime/CL/CLTunerTypes.h>

namespace armnn
{

struct GpuFsaBackendContext::GpuFsaContextControlWrapper
{
    GpuFsaContextControlWrapper(arm_compute::CLTuner* tuner,
                                arm_compute::CLGEMMHeuristicsHandle* heuristicsHandle,
                                bool profilingEnabled)
        : m_GpuFsaContextControl(tuner, heuristicsHandle, profilingEnabled)
    {}

    bool Sync()
    {
        if (arm_compute::CLScheduler::get().context()() != NULL)
        {
            // Waits for all queued CL requests to finish before unloading the network they may be using.
            try
            {
                // Coverity fix: arm_compute::CLScheduler::sync() may throw an exception of type cl::Error.
                arm_compute::CLScheduler::get().sync();
            }
            catch (const cl::Error& err)
            {
                ARMNN_LOG(warning) << "Runtime::UnloadNetwork(): an error occurred while waiting for "
                                      "the queued CL requests to finish";
                throw err;
            }
        }

        return true;
    }

    void ClearClCache()
    {
        if (arm_compute::CLScheduler::get().context()() != NULL)
        {
            // There are no loaded networks left, so clear the CL cache to free up memory
            m_GpuFsaContextControl.ClearClCache();
        }
    }

    GpuFsaContextControl m_GpuFsaContextControl;
};

GpuFsaBackendContext::GpuFsaBackendContext(const IRuntime::CreationOptions& options)
    : IBackendContext(options)
    , m_TuningFile()
{
    bool kernelProfiling = options.m_EnableGpuProfiling;

    arm_compute::CLTuner* tuner = nullptr;
    arm_compute::CLGEMMHeuristicsHandle* mlgoTuner = nullptr;
    bool useLegacyTunerAPI = options.m_GpuAccTunedParameters.get() != nullptr;
    if (useLegacyTunerAPI)
    {
        auto clTunerParams = PolymorphicDowncast<ClTunedParameters*>(
                                options.m_GpuAccTunedParameters.get());
        tuner = &clTunerParams->m_Tuner;

        if (tuner)
        {
            auto ConvertTuningLevel = [](IGpuAccTunedParameters::TuningLevel level,
                                         armnn::IGpuAccTunedParameters::Mode mode)
                {
                    if (mode == armnn::IGpuAccTunedParameters::Mode::UseTunedParameters)
                    {
                        return TuningLevel::None;
                    }

                    switch(level)
                    {
                        case IGpuAccTunedParameters::TuningLevel::Rapid:
                            return TuningLevel::Rapid;
                        case IGpuAccTunedParameters::TuningLevel::Normal:
                            return TuningLevel::Normal;
                        case IGpuAccTunedParameters::TuningLevel::Exhaustive:
                            return TuningLevel::Exhaustive;
                        default:
                        {
                            ARMNN_LOG(warning) << "Tuning level not recognised.";
                            return TuningLevel::None;
                        }
                    }
                };

            TuningLevel tuningLevel = ConvertTuningLevel(clTunerParams->m_TuningLevel, clTunerParams->m_Mode);
            ConfigureTuner(*tuner, tuningLevel);
        }
    }
    else //New backend options API
    {
        const TuningLevel defaultTuningLevel = TuningLevel::None;
        auto tuningLevel = defaultTuningLevel;

        ParseOptions(options.m_BackendOptions, "GpuFsa", [&](std::string name, const BackendOptions::Var& value)
            {
                if (name == "KernelProfilingEnabled")
                {
                    kernelProfiling |= ParseBooleanBackendOption(value, false);
                } else if (name == "TuningFile")
                {
                    m_TuningFile = ParseStringBackendOption(value, "");
                } else if (name == "TuningLevel")
                {
                    tuningLevel = ParseTuningLevel(value, defaultTuningLevel);
                }
                else if (name == "MLGOTuningFilePath")
                {
                    m_MLGOTuningFile = ParseStringBackendOption(value, "");
                }
            });

        // Create the tuner, in tuning mode initially.
        m_Tuner = std::make_unique<arm_compute::CLTuner>(true);

        ConfigureTuner(*(m_Tuner.get()), tuningLevel);

        if (!m_TuningFile.empty())
        {
            try
            {
                ARMNN_LOG(info) << "Loading Gpu tuning data from file: " << m_TuningFile;
                m_Tuner->load_from_file(m_TuningFile.c_str());
            }
            catch (const std::exception& e)
            {
                // Warn if not tuning, otherwise tuning will generate new params
                if (tuningLevel == TuningLevel::None)
                {
                    ARMNN_LOG(warning) << "Could not load GpuFsa tuner data file.";
                }
            }
        }

        if (!m_MLGOTuningFile.empty())
        {
            try
            {
                ARMNN_LOG(info) << "Loading Gpu MLGO tuning data from file: " << m_TuningFile;
                if(m_MLGOTuner.reload_from_file(m_MLGOTuningFile.c_str()))
                {
                    mlgoTuner = &m_MLGOTuner;
                }
            }
            catch (const std::exception& e)
            {
                ARMNN_LOG(warning) << "Could not load GpuFsa MLGO tuner data file.";
            }
        }

        tuner = m_Tuner.get();
    }

    m_GpuFsaContextControlWrapper = std::make_unique<GpuFsaContextControlWrapper>(
            tuner,
            mlgoTuner,
            kernelProfiling
    );
}

bool GpuFsaBackendContext::BeforeLoadNetwork(NetworkId)
{
    return true;
}

bool GpuFsaBackendContext::AfterLoadNetwork(NetworkId networkId)
{
    {
        std::lock_guard<std::mutex> lockGuard(m_Mutex);
        m_NetworkIds.insert(networkId);
    }
    return true;
}

bool GpuFsaBackendContext::BeforeUnloadNetwork(NetworkId)
{
    return m_GpuFsaContextControlWrapper->Sync();
}

bool GpuFsaBackendContext::AfterUnloadNetwork(NetworkId networkId)
{
    bool clearCache = false;
    {
        std::lock_guard<std::mutex> lockGuard(m_Mutex);
        m_NetworkIds.erase(networkId);
        clearCache = m_NetworkIds.empty();
    }

    if (clearCache)
    {
        m_GpuFsaContextControlWrapper->ClearClCache();
    }

    return true;
}

bool GpuFsaBackendContext::AfterEnqueueWorkload(NetworkId)
{
    return m_GpuFsaContextControlWrapper->Sync();
}

GpuFsaBackendContext::~GpuFsaBackendContext()
{
    if (m_Tuner && !m_TuningFile.empty())
    {
        try
        {
            m_Tuner->save_to_file(m_TuningFile.c_str());
        }
        catch(const std::exception& e)
        {
            ARMNN_LOG(warning) << "Could not save GpuFsa tuner data to file " << m_TuningFile;
        }
    }
}

} // namespace armnn