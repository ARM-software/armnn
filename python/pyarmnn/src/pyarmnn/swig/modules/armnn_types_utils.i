//
// Copyright © 2020 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
%{
#include "armnn/TypesUtils.hpp"
%}

namespace armnn
{
    constexpr unsigned int GetDataTypeSize(DataType dataType);

    constexpr const char* GetDataTypeName(DataType dataType);

    template<typename QuantizedType>
    QuantizedType Quantize(float value, float scale, int32_t offset);
    %template(Quantize_uint8_t) Quantize<uint8_t>;
    %template(Quantize_int8_t) Quantize<int8_t>;
    %template(Quantize_int16_t) Quantize<int16_t>;
    %template(Quantize_int32_t) Quantize<int32_t>;

    template <typename QuantizedType>
    float Dequantize(QuantizedType value, float scale, int32_t offset);
    %template(Dequantize_uint8_t) Dequantize<uint8_t>;
    %template(Dequantize_int8_t) Dequantize<int8_t>;
    %template(Dequantize_int16_t) Dequantize<int16_t>;
    %template(Dequantize_int32_t) Dequantize<int32_t>;
}