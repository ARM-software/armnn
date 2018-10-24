//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include "BackendId.hpp"

namespace armnn
{

class IBackend
{
protected:
    IBackend() {}
    virtual ~IBackend() {}

public:
    virtual const BackendId& GetId() const = 0;
};

} // namespace armnn