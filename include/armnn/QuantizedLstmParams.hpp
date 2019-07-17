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

    const ConstTensor& deref(const ConstTensor* tensorPtr) const
    {
        if (tensorPtr != nullptr)
        {
            const ConstTensor &temp = *tensorPtr;
            return temp;
        }
        throw InvalidArgumentException("QuantizedLstmInputParams: Can't dereference a null pointer");
    }

    const ConstTensor& get_InputToInputWeights() const
    {
        return deref(m_InputToInputWeights);
    }

    const ConstTensor& get_InputToForgetWeights() const
    {
        return deref(m_InputToForgetWeights);
    }

    const ConstTensor& get_InputToCellWeights() const
    {
        return deref(m_InputToCellWeights);
    }

    const ConstTensor& get_InputToOutputWeights() const
    {
        return deref(m_InputToOutputWeights);
    }

    const ConstTensor& get_RecurrentToInputWeights() const
    {
        return deref(m_RecurrentToInputWeights);
    }

    const ConstTensor& get_RecurrentToForgetWeights() const
    {
        return deref(m_RecurrentToForgetWeights);
    }

    const ConstTensor& get_RecurrentToCellWeights() const
    {
        return deref(m_RecurrentToCellWeights);
    }

    const ConstTensor& get_RecurrentToOutputWeights() const
    {
        return deref(m_RecurrentToOutputWeights);
    }

    const ConstTensor& get_InputGateBias() const
    {
        return deref(m_InputGateBias);
    }

    const ConstTensor& get_ForgetGateBias() const
    {
        return deref(m_ForgetGateBias);
    }

    const ConstTensor& get_CellBias() const
    {
        return deref(m_CellBias);
    }

    const ConstTensor& get_OutputGateBias() const
    {
        return deref(m_OutputGateBias);
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
};

} // namespace armnn

