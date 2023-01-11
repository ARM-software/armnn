//
// Copyright © 2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ArmComputeTuningUtils.hpp"

namespace armnn
{

IGpuAccTunedParameters* IGpuAccTunedParameters::CreateRaw(IGpuAccTunedParameters::Mode mode,
                                                          IGpuAccTunedParameters::TuningLevel tuningLevel)
{
    return new ClTunedParameters(mode, tuningLevel);
}

IGpuAccTunedParametersPtr IGpuAccTunedParameters::Create(IGpuAccTunedParameters::Mode mode,
                                                         IGpuAccTunedParameters::TuningLevel tuningLevel)
{
    return IGpuAccTunedParametersPtr(CreateRaw(mode, tuningLevel), &IGpuAccTunedParameters::Destroy);
}

void IGpuAccTunedParameters::Destroy(IGpuAccTunedParameters* params)
{
    delete params;
}

ClTunedParameters::ClTunedParameters(IGpuAccTunedParameters::Mode mode,
                                     IGpuAccTunedParameters::TuningLevel tuningLevel)
    : m_Mode(mode)
    , m_TuningLevel(tuningLevel)
    , m_Tuner(mode == ClTunedParameters::Mode::UpdateTunedParameters)
{
}

void ClTunedParameters::Load(const char* filename)
{
    try
    {
        m_Tuner.load_from_file(filename);
    }
    catch (const std::exception& e)
    {
        throw Exception(std::string("Failed to load tuned parameters file '") + filename + "': " + e.what());
    }
}

void ClTunedParameters::Save(const char* filename) const
{
    try
    {
        m_Tuner.save_to_file(filename);
    }
    catch (const std::exception& e)
    {
        throw Exception(std::string("Failed to save tuned parameters file to '") + filename + "': " + e.what());
    }
}

}