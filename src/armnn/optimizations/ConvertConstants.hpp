//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "Optimization.hpp"

#include <armnnUtils/FloatingPointConverter.hpp>

#include <backendsCommon/CpuTensorHandle.hpp>

#include <armnn/utility/IgnoreUnused.hpp>

#include <BFloat16.hpp>
#include <Half.hpp>

namespace armnn
{
namespace optimizations
{

struct BFloat16ToFloat32
{
    static void Func(std::unique_ptr<ScopedCpuTensorHandle>& handle)
    {
        const TensorInfo& info = handle->GetTensorInfo();

        if (info.GetDataType() == DataType::BFloat16)
        {
            std::vector<float> newValues(info.GetNumElements());

            armnnUtils::FloatingPointConverter::ConvertBFloat16ToFloat32(handle->GetTensor<BFloat16>(),
                                                                         info.GetNumElements(),
                                                                         newValues.data());

            TensorInfo newInfo(info.GetShape(), DataType::Float32);
            ConstTensor newInput(newInfo, newValues);
            handle.reset(new ScopedCpuTensorHandle(newInput));
        }
    }
};

struct Float16ToFloat32
{
    static void Func(std::unique_ptr<ScopedCpuTensorHandle>& handle)
    {
        const TensorInfo& info = handle->GetTensorInfo();

        if (info.GetDataType() == DataType::Float16)
        {
            std::vector<float> newValues(info.GetNumElements());

            armnnUtils::FloatingPointConverter::ConvertFloat16To32(handle->GetTensor<Half>(),
                                                                   info.GetNumElements(),
                                                                   newValues.data());

            TensorInfo newInfo(info.GetShape(), DataType::Float32);
            ConstTensor newInput(newInfo, newValues);
            handle.reset(new ScopedCpuTensorHandle(newInput));
        }
    }
};

struct Float32ToBFloat16
{
    static void Func(std::unique_ptr<ScopedCpuTensorHandle>& handle)
    {
        const TensorInfo& info = handle->GetTensorInfo();

        if (info.GetDataType() == DataType::Float32)
        {
            std::vector<BFloat16> newValues(info.GetNumElements());

            armnnUtils::FloatingPointConverter::ConvertFloat32ToBFloat16(handle->GetTensor<float>(),
                                                                         info.GetNumElements(),
                                                                         newValues.data());

            TensorInfo newInfo(info.GetShape(), DataType::BFloat16);
            ConstTensor newInput(newInfo, newValues);
            handle.reset(new ScopedCpuTensorHandle(newInput));
        }
    }
};

struct Float32ToFloat16
{
    static void Func(std::unique_ptr<ScopedCpuTensorHandle>& handle)
    {
        const TensorInfo& info = handle->GetTensorInfo();

        if (info.GetDataType() == DataType::Float32)
        {
            std::vector<Half> newValues(info.GetNumElements());

            armnnUtils::FloatingPointConverter::ConvertFloat32To16(handle->GetTensor<float>(),
                                                                   info.GetNumElements(),
                                                                   newValues.data());

            TensorInfo newInfo(info.GetShape(), DataType::Float16);
            ConstTensor newInput(newInfo, newValues);
            handle.reset(new ScopedCpuTensorHandle(newInput));
        }
    }
};

template<typename Converter, typename Predicate>
class ConvertConstants : public Optimization
{
public:
    ConvertConstants() = default;
    ConvertConstants(const ConvertConstants&) = default;
    virtual ~ConvertConstants() = default;

    void Run(Graph& graph, Layer& layer) const override
    {
        IgnoreUnused(graph);
        if (Predicate::Test(layer))
        {
            layer.OperateOnConstantTensors(Converter::Func);
        }
    }
protected:
};

struct IsFloat32Layer
{
    static bool Test(const Layer& layer)
    {
        return layer.GetDataType() == DataType::Float32;
    }
};

struct IsFloat16Layer
{
    static bool Test(const Layer& layer)
    {
        return layer.GetDataType() == DataType::Float16;
    }
};

struct IsBFloat16Layer
{
    static bool Test(const Layer& layer)
    {
        return layer.GetDataType() == DataType::BFloat16;
    }
};

using ConvertConstantsBFloatToFloat = ConvertConstants<BFloat16ToFloat32, IsFloat32Layer>;
using ConvertConstantsFloatToBFloat = ConvertConstants<Float32ToBFloat16, IsBFloat16Layer>;

using ConvertConstantsHalfToFloat = ConvertConstants<Float16ToFloat32, IsFloat32Layer>;
using ConvertConstantsFloatToHalf = ConvertConstants<Float32ToFloat16, IsFloat16Layer>;

} //namespace optimizations
} //namespace armnn
