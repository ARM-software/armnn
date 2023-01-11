//
// Copyright © 2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <armnn/BackendOptions.hpp>
#include <armnn/IRuntime.hpp>
#include <armnn/Logging.hpp>

#include <arm_compute/runtime/CL/CLTuner.h>
#include <arm_compute/runtime/CL/CLTunerTypes.h>
#include <arm_compute/runtime/CL/CLGEMMHeuristicsHandle.h>

namespace armnn
{

enum class TuningLevel
{
    None,
    Rapid,
    Normal,
    Exhaustive
};

inline TuningLevel ParseTuningLevel(const BackendOptions::Var& value, TuningLevel defaultValue)
{
    if (value.IsInt())
    {
        int v = value.AsInt();
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

inline void ConfigureTuner(arm_compute::CLTuner &tuner, TuningLevel level)
{
    tuner.set_tune_new_kernels(true); // Turn on tuning initially.

    switch (level)
    {
        case TuningLevel::Rapid:
            ARMNN_LOG(info) << "Gpu tuning is activated. TuningLevel: Rapid (1)";
            tuner.set_tuner_mode(arm_compute::CLTunerMode::RAPID);
            break;
        case TuningLevel::Normal:
            ARMNN_LOG(info) << "Gpu tuning is activated. TuningLevel: Normal (2)";
            tuner.set_tuner_mode(arm_compute::CLTunerMode::NORMAL);
            break;
        case TuningLevel::Exhaustive:
            ARMNN_LOG(info) << "Gpu tuning is activated. TuningLevel: Exhaustive (3)";
            tuner.set_tuner_mode(arm_compute::CLTunerMode::EXHAUSTIVE);
            break;
        case TuningLevel::None:
        default:
            tuner.set_tune_new_kernels(false); // Turn off tuning. Set to "use" only mode.
            break;
    }
}

class ClTunedParameters : public IGpuAccTunedParameters
{
public:
    ClTunedParameters(IGpuAccTunedParameters::Mode mode, IGpuAccTunedParameters::TuningLevel tuningLevel);

    virtual void Load(const char* filename);
    virtual void Save(const char* filename) const;

    Mode m_Mode;
    TuningLevel m_TuningLevel;

    arm_compute::CLTuner m_Tuner;
    arm_compute::CLGEMMHeuristicsHandle m_HeuristicsHandle;
};

}