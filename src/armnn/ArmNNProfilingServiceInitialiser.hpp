//
// Copyright © 2022 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <client/include/IInitialiseProfilingService.hpp>
#include <client/include/IProfilingService.hpp>

namespace armnn
{

class ArmNNProfilingServiceInitialiser : public arm::pipe::IInitialiseProfilingService
{
public:
    void InitialiseProfilingService(arm::pipe::IProfilingService& profilingService) override;
};

} // namespace armnn
