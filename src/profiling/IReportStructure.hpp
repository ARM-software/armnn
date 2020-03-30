//
// Copyright © 2020 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

namespace armnn
{

namespace profiling
{

class IReportStructure
{
public:
    virtual ~IReportStructure() {}
    virtual void ReportStructure() = 0;
};

} // namespace profiling

} // namespace armnn

