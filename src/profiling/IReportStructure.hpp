//
// Copyright © 2020 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

namespace arm
{

namespace pipe
{

class IProfilingService;

class IReportStructure
{
public:
    virtual ~IReportStructure() {}
    virtual void ReportStructure(arm::pipe::IProfilingService& profilingService) = 0;
};

} // namespace pipe

} // namespace arm

