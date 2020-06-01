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

    const TensorInfo& Deref(const TensorInfo* tensorInfo) const
    {
        if (tensorInfo != nullptr)
        {
            const TensorInfo &temp = *tensorInfo;
            return temp;
        }
        throw InvalidArgumentException("Can't dereference a null pointer");
    }

    const TensorInfo& GetInputToInputWeights() const
    {
        return Deref(m_InputToInputWeights);
    }
    const TensorInfo& GetInputToForgetWeights() const
    {
        return Deref(m_InputToForgetWeights);
    }
    const TensorInfo& GetInputToCellWeights() const
    {
        return Deref(m_InputToCellWeights);
    }
    const TensorInfo& GetInputToOutputWeights() const
    {
        return Deref(m_InputToOutputWeights);
    }
    const TensorInfo& GetRecurrentToInputWeights() const
    {
        return Deref(m_RecurrentToInputWeights);
    }
    const TensorInfo& GetRecurrentToForgetWeights() const
    {
        return Deref(m_RecurrentToForgetWeights);
    }
    const TensorInfo& GetRecurrentToCellWeights() const
    {
        return Deref(m_RecurrentToCellWeights);
    }
    const TensorInfo& GetRecurrentToOutputWeights() const
    {
        return Deref(m_RecurrentToOutputWeights);
    }
    const TensorInfo& GetCellToInputWeights() const
    {
        return Deref(m_CellToInputWeights);
    }
    const TensorInfo& GetCellToForgetWeights() const
    {
        return Deref(m_CellToForgetWeights);
    }
    const TensorInfo& GetCellToOutputWeights() const
    {
        return Deref(m_CellToOutputWeights);
    }
    const TensorInfo& GetInputGateBias() const
    {
        return Deref(m_InputGateBias);
    }
    const TensorInfo& GetForgetGateBias() const
    {
        return Deref(m_ForgetGateBias);
    }
    const TensorInfo& GetCellBias() const
    {
        return Deref(m_CellBias);
    }
    const TensorInfo& GetOutputGateBias() const
    {
        return Deref(m_OutputGateBias);
    }
    const TensorInfo& GetProjectionWeights() const
    {
        return Deref(m_ProjectionWeights);
    }
    const TensorInfo& GetProjectionBias() const
    {
        return Deref(m_ProjectionBias);
    }
    const TensorInfo& GetInputLayerNormWeights() const
    {
        return Deref(m_InputLayerNormWeights);
    }
    const TensorInfo& GetForgetLayerNormWeights() const
    {
        return Deref(m_ForgetLayerNormWeights);
    }
    const TensorInfo& GetCellLayerNormWeights() const
    {
        return Deref(m_CellLayerNormWeights);
    }
    const TensorInfo& GetOutputLayerNormWeights() const
    {
        return Deref(m_OutputLayerNormWeights);
    }
};

} // namespace armnn

