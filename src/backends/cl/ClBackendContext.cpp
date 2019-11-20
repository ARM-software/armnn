//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ClBackendContext.hpp"
#include "ClContextControl.hpp"

#include <armnn/Logging.hpp>

#include <arm_compute/core/CL/OpenCL.h>
#include <arm_compute/core/CL/CLKernelLibrary.h>
#include <arm_compute/runtime/CL/CLScheduler.h>
#include <arm_compute/runtime/CL/CLTunerTypes.h>

#include <boost/polymorphic_cast.hpp>

namespace armnn
{

struct ClBackendContext::ClContextControlWrapper
{
    ClContextControlWrapper(arm_compute::CLTuner* tuner,
                            bool profilingEnabled)
        : m_ClContextControl(tuner, profilingEnabled)
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
            catch (const cl::Error&)
            {
                ARMNN_LOG(warning) << "Runtime::UnloadNetwork(): an error occurred while waiting for "
                                      "the queued CL requests to finish";
                return false;
            }
        }

        return true;
    }

    void ClearClCache()
    {
        if (arm_compute::CLScheduler::get().context()() != NULL)
        {
            // There are no loaded networks left, so clear the CL cache to free up memory
            m_ClContextControl.ClearClCache();
        }
    }

    ClContextControl m_ClContextControl;
};

std::string LowerString(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c){ return std::tolower(c); });

    return value;
}

enum class TuningLevel
{
    None,
    Rapid,
    Normal,
    Exhaustive
};


TuningLevel ParseTuningLevel(const BackendOptions::Var& value, TuningLevel defaultValue)
{
    if (value.IsInt())
    {
        int v = value.IsInt();
        if (v > static_cast<int>(TuningLevel::Exhaustive) ||
            v < static_cast<int>(TuningLevel::None))
        {
            ARMNN_LOG(warning) << "Invalid GpuAcc tuning level ("<< v << ") selected. "
                                  "Using default(" << static_cast<int>(defaultValue) << ")";
        } else
        {
            return static_cast<TuningLevel>(v);
        }
    }
    return defaultValue;
}

bool ParseBoolean(const BackendOptions::Var& value, bool defaultValue)
{
    if (value.IsBool())
    {
        return value.AsBool();
    }

    return defaultValue;
}

std::string ParseFile(const BackendOptions::Var& value, std::string defaultValue)
{
    if (value.IsString())
    {
        return value.AsString();
    }
    return defaultValue;
}

template <typename F>
void ParseOptions(const std::vector<BackendOptions>& options, BackendId backend, F f)
{
    for (auto optionsGroup : options)
    {
        if (optionsGroup.GetBackendId() == backend)
        {
            for (size_t i=0; i < optionsGroup.GetOptionCount(); i++)
            {
                const BackendOptions::BackendOption option = optionsGroup.GetOption(i);
                f(option.GetName(), option.GetValue());
            }
        }
    }
}

ClBackendContext::ClBackendContext(const IRuntime::CreationOptions& options)
    : IBackendContext(options)
{
    bool kernelProfiling = options.m_EnableGpuProfiling;
    const TuningLevel defaultTuningLevel = TuningLevel::None;
    auto tuningLevel = defaultTuningLevel;
    m_TuningFile = "";


    arm_compute::CLTuner* tuner = nullptr;
    if (m_TuningFile.empty() == false)
    {
        bool useLegacyTunerAPI = options.m_GpuAccTunedParameters.get() != nullptr;
        if (useLegacyTunerAPI)
        {
            auto clTunerParams = boost::polymorphic_downcast<ClTunedParameters*>(
                                    options.m_GpuAccTunedParameters.get());
            auto clTuner = &clTunerParams->m_Tuner;

            if (clTuner)
            {
                auto ConvertTuningLevel = [](IGpuAccTunedParameters::TuningLevel level)
                    {
                        switch(level)
                        {
                            case IGpuAccTunedParameters::TuningLevel::Rapid:
                                return arm_compute::CLTunerMode::RAPID;
                            case IGpuAccTunedParameters::TuningLevel::Normal:
                                return arm_compute::CLTunerMode::NORMAL;
                            case IGpuAccTunedParameters::TuningLevel::Exhaustive:
                                return arm_compute::CLTunerMode::EXHAUSTIVE;
                            default:
                            {
                                BOOST_ASSERT_MSG(false, "Tuning level not recognised.");
                                return arm_compute::CLTunerMode::NORMAL;
                            }
                        }
                    };

                clTuner->set_tuner_mode(ConvertTuningLevel(clTunerParams->m_TuningLevel));
                clTuner->set_tune_new_kernels(
                    clTunerParams->m_Mode == armnn::IGpuAccTunedParameters::Mode::UpdateTunedParameters);
            }
        }
        else //New backend options API
        {
            ParseOptions(options.m_BackendOptions, "GpuAcc", [&](std::string name, const BackendOptions::Var& value)
                {
                    if (name == "KernelProfilingEnabled")
                    {
                        kernelProfiling |= ParseBoolean(value, false);
                    } else if (name == "TuningFile")
                    {
                        m_TuningFile = ParseFile(value, "");
                    } else if (name == "TuningLevel")
                    {
                        tuningLevel = ParseTuningLevel(value, defaultTuningLevel);
                    }
                });

            // Create the tuner, in tuning mode initially.
            m_Tuner = std::make_unique<arm_compute::CLTuner>(true);

            switch (tuningLevel)
            {
                case TuningLevel::Rapid:
                    m_Tuner->set_tuner_mode(arm_compute::CLTunerMode::RAPID);
                    break;
                case TuningLevel::Normal:
                    m_Tuner->set_tuner_mode(arm_compute::CLTunerMode::NORMAL);
                    break;
                case TuningLevel::Exhaustive:
                    m_Tuner->set_tuner_mode(arm_compute::CLTunerMode::EXHAUSTIVE);
                    break;
                case TuningLevel::None:
                default:
                    m_Tuner->set_tune_new_kernels(false); // Turn of tuning. Set to "use" only mode.
                    break;
            }

            try
            {
                m_Tuner->load_from_file(m_TuningFile.c_str());
            } catch (const std::exception& e)
            {
                ARMNN_LOG(warning) << "Could not load GpuAcc tuner data file.";
            }

            tuner = m_Tuner.get();
        }
    }

    m_ClContextControlWrapper = std::make_unique<ClContextControlWrapper>(
            tuner,
            kernelProfiling
        );
}

bool ClBackendContext::BeforeLoadNetwork(NetworkId)
{
    return true;
}

bool ClBackendContext::AfterLoadNetwork(NetworkId networkId)
{
    {
        std::lock_guard<std::mutex> lockGuard(m_Mutex);
        m_NetworkIds.insert(networkId);
    }
    return true;
}

bool ClBackendContext::BeforeUnloadNetwork(NetworkId)
{
    return m_ClContextControlWrapper->Sync();
}

bool ClBackendContext::AfterUnloadNetwork(NetworkId networkId)
{
    bool clearCache = false;
    {
        std::lock_guard<std::mutex> lockGuard(m_Mutex);
        m_NetworkIds.erase(networkId);
        clearCache = m_NetworkIds.empty();
    }

    if (clearCache)
    {
        m_ClContextControlWrapper->ClearClCache();
    }

    return true;
}

ClBackendContext::~ClBackendContext()
{
    if (m_Tuner && !m_TuningFile.empty())
    {
        try
        {
            m_Tuner->save_to_file(m_TuningFile.c_str());
        }
        catch(const std::exception& e)
        {
            ARMNN_LOG(warning) << "Could not save GpuAcc tuner data to file " << m_TuningFile;
        }
    }
}

} // namespace armnn