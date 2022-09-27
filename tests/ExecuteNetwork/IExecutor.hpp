//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
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
    virtual void CompareAndPrintResult(std::vector<const void*> otherOutput) = 0;
    virtual ~IExecutor(){};
    bool m_constructionFailed = false;
};
