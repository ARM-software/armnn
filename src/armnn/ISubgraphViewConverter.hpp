//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/Deprecated.hpp>

#include <memory>
#include <vector>
#include <functional>

namespace armnn
{

using CompiledBlobDeleter = std::function<void(const void*)>;
using CompiledBlobPtr = std::unique_ptr<void, CompiledBlobDeleter>;

class ISubgraphViewConverter
{
public:
    virtual ~ISubgraphViewConverter() {}

    virtual std::vector<CompiledBlobPtr> CompileNetwork() = 0;
};

} // namespace armnn
