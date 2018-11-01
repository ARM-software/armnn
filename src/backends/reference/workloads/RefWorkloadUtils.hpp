//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <backendsCommon/CpuTensorHandle.hpp>

#include <armnn/Tensor.hpp>
#include <armnn/Types.hpp>
#include <Half.hpp>

#include <boost/polymorphic_cast.hpp>

namespace armnn
{

////////////////////////////////////////////
/// float32 helpers
////////////////////////////////////////////

inline const TensorInfo& GetTensorInfo(const ITensorHandle* tensorHandle)
{
    // We know that reference workloads use CpuTensorHandles only, so this cast is legitimate.
    const ConstCpuTensorHandle* cpuTensorHandle =
        boost::polymorphic_downcast<const ConstCpuTensorHandle*>(tensorHandle);
    return cpuTensorHandle->GetTensorInfo();
}

template <typename DataType>
inline const DataType* GetConstCpuData(const ITensorHandle* tensorHandle)
{
    // We know that reference workloads use (Const)CpuTensorHandles only, so this cast is legitimate.
    const ConstCpuTensorHandle* cpuTensorHandle =
        boost::polymorphic_downcast<const ConstCpuTensorHandle*>(tensorHandle);
    return cpuTensorHandle->GetConstTensor<DataType>();
}

template <typename DataType>
inline DataType* GetCpuData(const ITensorHandle* tensorHandle)
{
    // We know that reference workloads use CpuTensorHandles only, so this cast is legitimate.
    const CpuTensorHandle* cpuTensorHandle = boost::polymorphic_downcast<const CpuTensorHandle*>(tensorHandle);
    return cpuTensorHandle->GetTensor<DataType>();
};

template <typename DataType, typename PayloadType>
const DataType* GetInputTensorData(unsigned int idx, const PayloadType& data)
{
    const ITensorHandle* tensorHandle = data.m_Inputs[idx];
    return GetConstCpuData<DataType>(tensorHandle);
}

template <typename DataType, typename PayloadType>
DataType* GetOutputTensorData(unsigned int idx, const PayloadType& data)
{
    const ITensorHandle* tensorHandle = data.m_Outputs[idx];
    return GetCpuData<DataType>(tensorHandle);
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

inline const uint8_t* GetConstCpuU8Data(const ITensorHandle* tensorHandle)
{
    // We know that reference workloads use (Const)CpuTensorHandles only, so this cast is legitimate.
    const ConstCpuTensorHandle* cpuTensorHandle =
        boost::polymorphic_downcast<const ConstCpuTensorHandle*>(tensorHandle);
    return cpuTensorHandle->GetConstTensor<uint8_t>();
};

inline uint8_t* GetCpuU8Data(const ITensorHandle* tensorHandle)
{
    // We know that reference workloads use CpuTensorHandles only, so this cast is legitimate.
    const CpuTensorHandle* cpuTensorHandle = boost::polymorphic_downcast<const CpuTensorHandle*>(tensorHandle);
    return cpuTensorHandle->GetTensor<uint8_t>();
};

template <typename PayloadType>
const uint8_t* GetInputTensorDataU8(unsigned int idx, const PayloadType& data)
{
    const ITensorHandle* tensorHandle = data.m_Inputs[idx];
    return GetConstCpuU8Data(tensorHandle);
}

template <typename PayloadType>
uint8_t* GetOutputTensorDataU8(unsigned int idx, const PayloadType& data)
{
    const ITensorHandle* tensorHandle = data.m_Outputs[idx];
    return GetCpuU8Data(tensorHandle);
}

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

inline void Quantize(uint8_t* quant, const float* dequant, const TensorInfo& info)
{
    for (size_t i = 0; i < info.GetNumElements(); i++)
    {
        quant[i] = armnn::Quantize<uint8_t>(dequant[i], info.GetQuantizationScale(), info.GetQuantizationOffset());
    }
}

} //namespace armnn
