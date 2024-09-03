//
// Copyright © 2022, 2024 Arm Ltd and Contributors.
// SPDX-License-Identifier: MIT
//

#pragma once
#include <vector>

/// IExecutor executes a network
class IExecutor
{
public:
    /// Execute the given network
    /// @return std::vector<const void*> A type erased vector of the outputs,
    /// that can be compared with the output of another IExecutor
    virtual std::vector<const void*> Execute()  = 0;
    /// Print available information about the network
    virtual void PrintNetworkInfo() = 0;
    /// Compare the output with the result of another IExecutor
    /// @return 0 if all output tensors match otherwise, the whole part of a non zero RMSE result.
    virtual unsigned int CompareAndPrintResult(std::vector<const void*> otherOutput) = 0;
    virtual ~IExecutor(){};
    bool m_constructionFailed = false;
};
