//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <armnn/backends/IBackendContext.hpp>

namespace armnn
{

/// The NeonBackendModelContext is used to pass in Neon specific backend ModelOptions. The supported backend
/// ModelOptions are:
///  - "FastMathEnabled"\n
///    Using the fast_math flag can lead to performance improvements in fp32 and fp16 layers but may result in\n
///    results with reduced or different precision. The fast_math flag will not have any effect on int8 performance.
///  - "NumberOfThreads"\n
///    Specify the number of threads used by the CpuAcc backend.
class NeonBackendModelContext : public IBackendModelContext
{
public:
    NeonBackendModelContext(const ModelOptions& modelOptions);

    bool IsFastMathEnabled() const;

    unsigned int GetNumberOfThreads() const;

private:
    bool m_IsFastMathEnabled;
    unsigned int m_NumberOfThreads;
};

} // namespace armnn