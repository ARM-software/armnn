//
// Copyright © 2020 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

namespace arm
{

namespace pipe
{

class INotifyBackends
{
public:
    virtual ~INotifyBackends() {}
    virtual void NotifyBackendsForTimelineReporting() = 0;
};

} // namespace pipe

} // namespace arm

