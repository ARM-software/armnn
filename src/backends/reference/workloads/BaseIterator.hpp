//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/ArmNN.hpp>
#include <TypeUtils.hpp>

namespace armnn
{

class BaseIterator
{
public:
    BaseIterator() {}

    virtual ~BaseIterator() {}

    virtual BaseIterator& operator++() = 0;

    virtual BaseIterator& operator+=(const unsigned int increment) = 0;

    virtual BaseIterator& operator-=(const unsigned int increment) = 0;
};

class Decoder : public BaseIterator
{
public:
    Decoder() : BaseIterator() {}

    virtual ~Decoder() {}

    virtual float Get() const = 0;
};

class Encoder : public BaseIterator
{
public:
    Encoder() : BaseIterator() {}

    virtual ~Encoder() {}

    virtual void Set(const float& right) = 0;
};

class ComparisonEncoder : public BaseIterator
{
public:
    ComparisonEncoder() : BaseIterator() {}

    virtual ~ComparisonEncoder() {}

    virtual void Set(bool right) = 0;
};

template<typename T, typename Base>
class TypedIterator : public Base
{
public:
    TypedIterator(T* data)
        : m_Iterator(data)
    {}

    TypedIterator& operator++() override
    {
        ++m_Iterator;
        return *this;
    }

    TypedIterator& operator+=(const unsigned int increment) override
    {
        m_Iterator += increment;
        return *this;
    }

    TypedIterator& operator-=(const unsigned int increment) override
    {
        m_Iterator -= increment;
        return *this;
    }

    T* m_Iterator;
};

class QASymm8Decoder : public TypedIterator<const uint8_t, Decoder>
{
public:
    QASymm8Decoder(const uint8_t* data, const float scale, const int32_t offset)
        : TypedIterator(data), m_Scale(scale), m_Offset(offset) {}

    float Get() const override
    {
        return armnn::Dequantize(*m_Iterator, m_Scale, m_Offset);
    }

private:
    const float m_Scale;
    const int32_t m_Offset;
};

class FloatDecoder : public TypedIterator<const float, Decoder>
{
public:
    FloatDecoder(const float* data)
        : TypedIterator(data) {}

    float Get() const override
    {
        return *m_Iterator;
    }
};

class QSymm16Decoder : public TypedIterator<const int16_t, Decoder>
{
public:
    QSymm16Decoder(const int16_t* data, const float scale, const int32_t offset)
        : TypedIterator(data), m_Scale(scale), m_Offset(offset) {}

    float Get() const override
    {
        return armnn::Dequantize(*m_Iterator, m_Scale, m_Offset);
    }

private:
    const float m_Scale;
    const int32_t m_Offset;
};

class FloatEncoder : public TypedIterator<float, Encoder>
{
public:
    FloatEncoder(float* data)
        : TypedIterator(data) {}

    void Set(const float& right) override
    {
        *m_Iterator = right;
    }
};

class QASymm8Encoder : public TypedIterator<uint8_t, Encoder>
{
public:
    QASymm8Encoder(uint8_t* data, const float scale, const int32_t offset)
        : TypedIterator(data), m_Scale(scale), m_Offset(offset) {}

    void Set(const float& right) override
    {
        *m_Iterator = armnn::Quantize<uint8_t>(right, m_Scale, m_Offset);
    }

private:
    const float m_Scale;
    const int32_t m_Offset;
};

class BooleanEncoder : public TypedIterator<uint8_t, ComparisonEncoder>
{
public:
    BooleanEncoder(uint8_t* data)
        : TypedIterator(data) {}

    void Set(bool right) override
    {
        *m_Iterator = right;
    }
};

class QSymm16Encoder : public TypedIterator<int16_t, Encoder>
{
public:
    QSymm16Encoder(int16_t* data, const float scale, const int32_t offset)
        : TypedIterator(data), m_Scale(scale), m_Offset(offset) {}

    void Set(const float& right) override
    {
        *m_Iterator = armnn::Quantize<int16_t>(right, m_Scale, m_Offset);
    }

private:
    const float m_Scale;
    const int32_t m_Offset;
};

} //namespace armnn