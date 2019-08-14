//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include "TensorFwd.hpp"
#include "Exceptions.hpp"

namespace armnn
{

struct QuantizedLstmInputParams
{
    QuantizedLstmInputParams()
        : m_InputToInputWeights(nullptr)
        , m_InputToForgetWeights(nullptr)
        , m_InputToCellWeights(nullptr)
        , m_InputToOutputWeights(nullptr)

        , m_RecurrentToInputWeights(nullptr)
        , m_RecurrentToForgetWeights(nullptr)
        , m_RecurrentToCellWeights(nullptr)
        , m_RecurrentToOutputWeights(nullptr)

        , m_InputGateBias(nullptr)
        , m_ForgetGateBias(nullptr)
        , m_CellBias(nullptr)
        , m_OutputGateBias(nullptr)
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

    const ConstTensor* m_InputGateBias;
    const ConstTensor* m_ForgetGateBias;
    const ConstTensor* m_CellBias;
    const ConstTensor* m_OutputGateBias;

    const ConstTensor& Deref(const ConstTensor* tensorPtr) const
    {
        if (tensorPtr != nullptr)
        {
            const ConstTensor &temp = *tensorPtr;
            return temp;
        }
        throw InvalidArgumentException("QuantizedLstmInputParams: Can't dereference a null pointer");
    }

    const ConstTensor& GetInputToInputWeights() const
    {
        return Deref(m_InputToInputWeights);
    }

    const ConstTensor& GetInputToForgetWeights() const
    {
        return Deref(m_InputToForgetWeights);
    }

    const ConstTensor& GetInputToCellWeights() const
    {
        return Deref(m_InputToCellWeights);
    }

    const ConstTensor& GetInputToOutputWeights() const
    {
        return Deref(m_InputToOutputWeights);
    }

    const ConstTensor& GetRecurrentToInputWeights() const
    {
        return Deref(m_RecurrentToInputWeights);
    }

    const ConstTensor& GetRecurrentToForgetWeights() const
    {
        return Deref(m_RecurrentToForgetWeights);
    }

    const ConstTensor& GetRecurrentToCellWeights() const
    {
        return Deref(m_RecurrentToCellWeights);
    }

    const ConstTensor& GetRecurrentToOutputWeights() const
    {
        return Deref(m_RecurrentToOutputWeights);
    }

    const ConstTensor& GetInputGateBias() const
    {
        return Deref(m_InputGateBias);
    }

    const ConstTensor& GetForgetGateBias() const
    {
        return Deref(m_ForgetGateBias);
    }

    const ConstTensor& GetCellBias() const
    {
        return Deref(m_CellBias);
    }

    const ConstTensor& GetOutputGateBias() const
    {
        return Deref(m_OutputGateBias);
    }
};

struct QuantizedLstmInputParamsInfo
{
    QuantizedLstmInputParamsInfo()
        : m_InputToInputWeights(nullptr)
        , m_InputToForgetWeights(nullptr)
        , m_InputToCellWeights(nullptr)
        , m_InputToOutputWeights(nullptr)

        , m_RecurrentToInputWeights(nullptr)
        , m_RecurrentToForgetWeights(nullptr)
        , m_RecurrentToCellWeights(nullptr)
        , m_RecurrentToOutputWeights(nullptr)

        , m_InputGateBias(nullptr)
        , m_ForgetGateBias(nullptr)
        , m_CellBias(nullptr)
        , m_OutputGateBias(nullptr)
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

    const TensorInfo* m_InputGateBias;
    const TensorInfo* m_ForgetGateBias;
    const TensorInfo* m_CellBias;
    const TensorInfo* m_OutputGateBias;


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
};

} // namespace armnn

