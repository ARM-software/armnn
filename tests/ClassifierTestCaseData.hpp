//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <vector>

template <typename DataType>
class ClassifierTestCaseData
{
public:
    ClassifierTestCaseData(unsigned int label, std::vector<DataType> inputImage)
     : m_Label(label)
     , m_InputImage(std::move(inputImage))
    {
    }

    const unsigned int m_Label;
    std::vector<DataType> m_InputImage;
};
