//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <armnn/backends/IBackendContext.hpp>

namespace armnn
{

class ClBackendModelContext : public IBackendModelContext
{
public:
    ClBackendModelContext(const ModelOptions& modelOptions);

    bool IsFastMathEnabled() const;

private:
    bool m_IsFastMathEnabled;
};

} // namespace armnn