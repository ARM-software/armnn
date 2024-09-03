//
// Copyright © 2023, 2024 Arm Ltd and Contributors.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "ExecuteNetworkProgramOptions.hpp"
#include "IExecutor.hpp"

class FileComparisonExecutor : public IExecutor
{
public:
    FileComparisonExecutor(const ExecuteNetworkParams& params);
    ~FileComparisonExecutor();
    std::vector<const void*> Execute() override;
    void PrintNetworkInfo() override;
    unsigned int CompareAndPrintResult(std::vector<const void*> otherOutput) override;

private:
    // Disallow copy and assignment constructors.
    FileComparisonExecutor(FileComparisonExecutor&);
    FileComparisonExecutor operator=(const FileComparisonExecutor&);

    ExecuteNetworkParams m_Params;
    std::vector<armnn::OutputTensors> m_OutputTensorsVec;
};
