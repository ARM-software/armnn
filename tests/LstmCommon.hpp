//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <string>
#include <utility>

namespace
{

struct LstmInput
{
    LstmInput(const std::vector<float>& inputSeq,
              const std::vector<float>& stateC,
              const std::vector<float>& stateH)
            : m_InputSeq(inputSeq)
            , m_StateC(stateC)
            , m_StateH(stateH)
    {}

    std::vector<float>        m_InputSeq;
    std::vector<float>        m_StateC;
    std::vector<float>        m_StateH;
};

using LstmInputs = std::pair<std::string, std::vector<LstmInput>>;

} // anonymous namespace