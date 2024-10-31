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
              const std::vector<float>& stateH,
              const std::vector<float>& stateC)
            : m_InputSeq(inputSeq)
            , m_StateH(stateH)
            , m_StateC(stateC)
    {}

    std::vector<float>        m_InputSeq;
    std::vector<float>        m_StateH;
    std::vector<float>        m_StateC;
};

using LstmInputs = std::pair<std::string, std::vector<LstmInput>>;

} // anonymous namespace