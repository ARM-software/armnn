//
// Copyright © 2020 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

namespace arm
{

namespace pipe
{

class IReportStructure
{
public:
    virtual ~IReportStructure() {}
    virtual void ReportStructure() = 0;
};

} // namespace pipe

} // namespace arm

