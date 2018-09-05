//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include "TensorFwd.hpp"

namespace armnn
{

struct LstmInputParams
{
    LstmInputParams()
        : m_InputToInputWeights(nullptr)
        , m_InputToForgetWeights(nullptr)
        , m_InputToCellWeights(nullptr)
        , m_InputToOutputWeights(nullptr)
        , m_RecurrentToInputWeights(nullptr)
        , m_RecurrentToForgetWeights(nullptr)
        , m_RecurrentToCellWeights(nullptr)
        , m_RecurrentToOutputWeights(nullptr)
        , m_CellToInputWeights(nullptr)
        , m_CellToForgetWeights(nullptr)
        , m_CellToOutputWeights(nullptr)
        , m_InputGateBias(nullptr)
        , m_ForgetGateBias(nullptr)
        , m_CellBias(nullptr)
        , m_OutputGateBias(nullptr)
        , m_ProjectionWeights(nullptr)
        , m_ProjectionBias(nullptr)
    {
    }

    const ConstTensor* m_InputToInputWeights;
    const ConstTensor* m_InputToForgetWeights;
    const ConstTensor* m_InputToCellWeights;
    const ConstTensor* m_InputToOutputWeights;
    const ConstTensor* m_RecurrentToInputWeights;
    const ConstTensor* m_RecurrentToForgetWeights;
    const ConstTensor* m_RecurrentToCellWeights;
    const ConstTensor* m_RecurrentToOutputWeights;
    const ConstTensor* m_CellToInputWeights;
    const ConstTensor* m_CellToForgetWeights;
    const ConstTensor* m_CellToOutputWeights;
    const ConstTensor* m_InputGateBias;
    const ConstTensor* m_ForgetGateBias;
    const ConstTensor* m_CellBias;
    const ConstTensor* m_OutputGateBias;
    const ConstTensor* m_ProjectionWeights;
    const ConstTensor* m_ProjectionBias;
};

} // namespace armnn

