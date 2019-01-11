//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <memory>

namespace armnn
{

class ISubGraphConverter
{
public:
    virtual ~ISubGraphConverter() {};

    virtual std::shared_ptr<void> GetOutput() = 0;
};

}

