//
// Copyright © 2020 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

namespace armnn
{

namespace profiling
{

class INotifyBackends
{
public:
    virtual ~INotifyBackends() {}
    virtual void NotifyBackendsForTimelineReporting() = 0;
};

} // namespace profiling

} // namespace armnn

