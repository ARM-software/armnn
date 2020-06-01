//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include "ClassifierTestCaseData.hpp"

#include <string>
#include <memory>

class Cifar10Database
{
public:
    using DataType = float;
    using TTestCaseData = ClassifierTestCaseData<DataType>;

    explicit Cifar10Database(const std::string& binaryFileDirectory, bool rgbPack = false);
    std::unique_ptr<TTestCaseData> GetTestCaseData(unsigned int testCaseId);

private:
    std::string m_BinaryDirectory;
    bool m_RgbPack;
};
