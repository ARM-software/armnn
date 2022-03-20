//
// Copyright © 2022 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

namespace arm
{

namespace pipe
{

// forward declaration
class IProfilingService;

class IInitialiseProfilingService
{
public:
    virtual ~IInitialiseProfilingService() {}
    virtual void InitialiseProfilingService(IProfilingService& profilingService) = 0;
};

} // namespace pipe

} // namespace arm
