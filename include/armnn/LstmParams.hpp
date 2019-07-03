//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include "TensorFwd.hpp"
#include "Exceptions.hpp"

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
        , m_InputLayerNormWeights(nullptr)
        , m_ForgetLayerNormWeights(nullptr)
        , m_CellLayerNormWeights(nullptr)
        , m_OutputLayerNormWeights(nullptr)
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
    const ConstTensor* m_InputLayerNormWeights;
    const ConstTensor* m_ForgetLayerNormWeights;
    const ConstTensor* m_CellLayerNormWeights;
    const ConstTensor* m_OutputLayerNormWeights;
};

struct LstmInputParamsInfo
{
    LstmInputParamsInfo()
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
            , m_InputLayerNormWeights(nullptr)
            , m_ForgetLayerNormWeights(nullptr)
            , m_CellLayerNormWeights(nullptr)
            , m_OutputLayerNormWeights(nullptr)
    {
    }
    const TensorInfo* m_InputToInputWeights;
    const TensorInfo* m_InputToForgetWeights;
    const TensorInfo* m_InputToCellWeights;
    const TensorInfo* m_InputToOutputWeights;
    const TensorInfo* m_RecurrentToInputWeights;
    const TensorInfo* m_RecurrentToForgetWeights;
    const TensorInfo* m_RecurrentToCellWeights;
    const TensorInfo* m_RecurrentToOutputWeights;
    const TensorInfo* m_CellToInputWeights;
    const TensorInfo* m_CellToForgetWeights;
    const TensorInfo* m_CellToOutputWeights;
    const TensorInfo* m_InputGateBias;
    const TensorInfo* m_ForgetGateBias;
    const TensorInfo* m_CellBias;
    const TensorInfo* m_OutputGateBias;
    const TensorInfo* m_ProjectionWeights;
    const TensorInfo* m_ProjectionBias;
    const TensorInfo* m_InputLayerNormWeights;
    const TensorInfo* m_ForgetLayerNormWeights;
    const TensorInfo* m_CellLayerNormWeights;
    const TensorInfo* m_OutputLayerNormWeights;

    const TensorInfo& deref(const TensorInfo* tensorInfo) const
    {
        if (tensorInfo != nullptr)
        {
            const TensorInfo &temp = *tensorInfo;
            return temp;
        }
        throw InvalidArgumentException("Can't dereference a null pointer");
    }

    const TensorInfo& get_InputToInputWeights() const
    {
        return deref(m_InputToInputWeights);
    }
    const TensorInfo& get_InputToForgetWeights() const
    {
        return deref(m_InputToForgetWeights);
    }
    const TensorInfo& get_InputToCellWeights() const
    {
        return deref(m_InputToCellWeights);
    }
    const TensorInfo& get_InputToOutputWeights() const
    {
        return deref(m_InputToOutputWeights);
    }
    const TensorInfo& get_RecurrentToInputWeights() const
    {
        return deref(m_RecurrentToInputWeights);
    }
    const TensorInfo& get_RecurrentToForgetWeights() const
    {
        return deref(m_RecurrentToForgetWeights);
    }
    const TensorInfo& get_RecurrentToCellWeights() const
    {
        return deref(m_RecurrentToCellWeights);
    }
    const TensorInfo& get_RecurrentToOutputWeights() const
    {
        return deref(m_RecurrentToOutputWeights);
    }
    const TensorInfo& get_CellToInputWeights() const
    {
        return deref(m_CellToInputWeights);
    }
    const TensorInfo& get_CellToForgetWeights() const
    {
        return deref(m_CellToForgetWeights);
    }
    const TensorInfo& get_CellToOutputWeights() const
    {
        return deref(m_CellToOutputWeights);
    }
    const TensorInfo& get_InputGateBias() const
    {
        return deref(m_InputGateBias);
    }
    const TensorInfo& get_ForgetGateBias() const
    {
        return deref(m_ForgetGateBias);
    }
    const TensorInfo& get_CellBias() const
    {
        return deref(m_CellBias);
    }
    const TensorInfo& get_OutputGateBias() const
    {
        return deref(m_OutputGateBias);
    }
    const TensorInfo& get_ProjectionWeights() const
    {
        return deref(m_ProjectionWeights);
    }
    const TensorInfo& get_ProjectionBias() const
    {
        return deref(m_ProjectionBias);
    }
    const TensorInfo& get_InputLayerNormWeights() const
    {
        return deref(m_InputLayerNormWeights);
    }
    const TensorInfo& get_ForgetLayerNormWeights() const
    {
        return deref(m_ForgetLayerNormWeights);
    }
    const TensorInfo& get_CellLayerNormWeights() const
    {
        return deref(m_CellLayerNormWeights);
    }
    const TensorInfo& get_OutputLayerNormWeights() const
    {
        return deref(m_OutputLayerNormWeights);
    }
};

} // namespace armnn

