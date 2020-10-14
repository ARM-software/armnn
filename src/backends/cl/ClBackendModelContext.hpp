//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <armnn/backends/IBackendContext.hpp>

namespace armnn
{

/// The ClBackendModelContext is used to pass in CL specific backend ModelOptions. The supported backend ModelOptions
/// are:
///  - "FastMathEnabled"\n
///    Using the fast_math flag can lead to performance improvements in fp32 and fp16 layers but may result in\n
///    results with reduced or different precision. The fast_math flag will not have any effect on int8 performance.
class ClBackendModelContext : public IBackendModelContext
{
public:
    ClBackendModelContext(const ModelOptions& modelOptions);

    bool IsFastMathEnabled() const;

private:
    bool m_IsFastMathEnabled;
};

} // namespace armnn