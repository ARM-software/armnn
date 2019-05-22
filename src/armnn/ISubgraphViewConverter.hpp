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

///
/// Old ISubGraphConverter definition kept for backward compatibility only.
///
using ISubGraphConverter ARMNN_DEPRECATED_MSG("This type is no longer supported") = ISubgraphViewConverter;

} // namespace armnn
