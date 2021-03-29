//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/backends/ITensorHandle.hpp>

#include <vector>

namespace armnn
{

namespace experimental
{

struct WorkingMemDescriptor
{
    std::vector<ITensorHandle*> m_Inputs;
    std::vector<ITensorHandle*> m_Outputs;

    ~WorkingMemDescriptor() = default;
    WorkingMemDescriptor() = default;
};

} // end experimental namespace

} // end armnn namespace
