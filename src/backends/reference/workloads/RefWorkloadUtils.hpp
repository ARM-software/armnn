//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <backendsCommon/CpuTensorHandle.hpp>

#include <armnn/Tensor.hpp>
#include <armnn/Types.hpp>

#include <reference/RefTensorHandle.hpp>

#include <Half.hpp>
#include <boost/polymorphic_cast.hpp>

namespace armnn
{

////////////////////////////////////////////
/// float32 helpers
////////////////////////////////////////////

inline const TensorInfo& GetTensorInfo(const ITensorHandle* tensorHandle)
{
    // We know that reference workloads use RefTensorHandles for inputs and outputs
    const RefTensorHandle* refTensorHandle =
        boost::polymorphic_downcast<const RefTensorHandle*>(tensorHandle);
    return refTensorHandle->GetTensorInfo();
}

template <typename DataType, typename PayloadType>
const DataType* GetInputTensorData(unsigned int idx, const PayloadType& data)
{
    const ITensorHandle* tensorHandle = data.m_Inputs[idx];
    return reinterpret_cast<const DataType*>(tensorHandle->Map());
}

template <typename DataType, typename PayloadType>
DataType* GetOutputTensorData(unsigned int idx, const PayloadType& data)
{
    ITensorHandle* tensorHandle = data.m_Outputs[idx];
    return reinterpret_cast<DataType*>(tensorHandle->Map());
}

template <typename PayloadType>
const float* GetInputTensorDataFloat(unsigned int idx, const PayloadType& data)
{
    return GetInputTensorData<float>(idx, data);
}

template <typename PayloadType>
float* GetOutputTensorDataFloat(unsigned int idx, const PayloadType& data)
{
    return GetOutputTensorData<float>(idx, data);
}

template <typename PayloadType>
const Half* GetInputTensorDataHalf(unsigned int idx, const PayloadType& data)
{
    return GetInputTensorData<Half>(idx, data);
}

template <typename PayloadType>
Half* GetOutputTensorDataHalf(unsigned int idx, const PayloadType& data)
{
    return GetOutputTensorData<Half>(idx, data);
}

////////////////////////////////////////////
/// u8 helpers
////////////////////////////////////////////

template<typename T>
std::vector<float> Dequantize(const T* quant, const TensorInfo& info)
{
    std::vector<float> ret(info.GetNumElements());
    for (size_t i = 0; i < info.GetNumElements(); i++)
    {
        ret[i] = armnn::Dequantize(quant[i], info.GetQuantizationScale(), info.GetQuantizationOffset());
    }
    return ret;
}

template<typename T>
inline void Dequantize(const T* inputData, float* outputData, const TensorInfo& info)
{
    for (unsigned int i = 0; i < info.GetNumElements(); i++)
    {
        outputData[i] = Dequantize<T>(inputData[i], info.GetQuantizationScale(), info.GetQuantizationOffset());
    }
}

inline void Quantize(uint8_t* quant, const float* dequant, const TensorInfo& info)
{
    for (size_t i = 0; i < info.GetNumElements(); i++)
    {
        quant[i] = armnn::Quantize<uint8_t>(dequant[i], info.GetQuantizationScale(), info.GetQuantizationOffset());
    }
}

} //namespace armnn
