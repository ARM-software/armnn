//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <memory>
#include <vector>
#include <functional>

namespace armnn
{

using CompiledBlobDeleter = std::function<void(const void*)>;
using CompiledBlobPtr = std::unique_ptr<void, CompiledBlobDeleter>;

class ISubGraphConverter
{
public:
    virtual ~ISubGraphConverter() {}

    virtual std::vector<CompiledBlobPtr> GetOutput() = 0;
};

} // namespace armnn
