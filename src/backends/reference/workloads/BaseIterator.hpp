//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/ArmNN.hpp>
#include <ResolveType.hpp>

#include <boost/assert.hpp>

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

    virtual BaseIterator& operator[](const unsigned int index) = 0;
};

template<typename IType>
class Decoder : public BaseIterator
{
public:
    Decoder() {}

    virtual ~Decoder() {}

    virtual void Reset(void*) = 0;

    virtual IType Get() const = 0;
};

template<typename IType>
class Encoder : public BaseIterator
{
public:
    Encoder() {}

    virtual ~Encoder() {}

    virtual void Reset(void*) = 0;

    virtual void Set(IType right) = 0;

    virtual IType Get() const = 0;
};

template<typename T, typename Base>
class TypedIterator : public Base
{
public:
    TypedIterator(T* data = nullptr)
        : m_Iterator(data), m_Start(data)
    {}

    void Reset(void* data) override
    {
        m_Iterator = reinterpret_cast<T*>(data);
        m_Start = m_Iterator;
    }

    TypedIterator& operator++() override
    {
        BOOST_ASSERT(m_Iterator);
        ++m_Iterator;
        return *this;
    }

    TypedIterator& operator+=(const unsigned int increment) override
    {
        BOOST_ASSERT(m_Iterator);
        m_Iterator += increment;
        return *this;
    }

    TypedIterator& operator-=(const unsigned int increment) override
    {
        BOOST_ASSERT(m_Iterator);
        m_Iterator -= increment;
        return *this;
    }

    TypedIterator& operator[](const unsigned int index) override
    {
        BOOST_ASSERT(m_Iterator);
        m_Iterator = m_Start + index;
        return *this;
    }

protected:
    T* m_Iterator;
    T* m_Start;
};

class QASymm8Decoder : public TypedIterator<const uint8_t, Decoder<float>>
{
public:
    QASymm8Decoder(const uint8_t* data, const float scale, const int32_t offset)
        : TypedIterator(data), m_Scale(scale), m_Offset(offset) {}

    QASymm8Decoder(const float scale, const int32_t offset)
        : QASymm8Decoder(nullptr, scale, offset) {}

    float Get() const override
    {
        return armnn::Dequantize(*m_Iterator, m_Scale, m_Offset);
    }

private:
    const float m_Scale;
    const int32_t m_Offset;
};

class QSymm16Decoder : public TypedIterator<const int16_t, Decoder<float>>
{
public:
    QSymm16Decoder(const int16_t* data, const float scale, const int32_t offset)
        : TypedIterator(data), m_Scale(scale), m_Offset(offset) {}

    QSymm16Decoder(const float scale, const int32_t offset)
        : QSymm16Decoder(nullptr, scale, offset) {}

    float Get() const override
    {
        return armnn::Dequantize(*m_Iterator, m_Scale, m_Offset);
    }

private:
    const float m_Scale;
    const int32_t m_Offset;
};

class FloatDecoder : public TypedIterator<const float, Decoder<float>>
{
public:
    FloatDecoder(const float* data)
        : TypedIterator(data) {}

    FloatDecoder()
        : FloatDecoder(nullptr) {}

    float Get() const override
    {
        return *m_Iterator;
    }
};

class ScaledInt32Decoder : public TypedIterator<const int32_t, Decoder<float>>
{
public:
    ScaledInt32Decoder(const int32_t* data, const float scale)
        : TypedIterator(data), m_Scale(scale) {}

    ScaledInt32Decoder(const float scale)
        : ScaledInt32Decoder(nullptr, scale) {}

    float Get() const override
    {
        return static_cast<float>(*m_Iterator) * m_Scale;
    }

private:
    const float m_Scale;
};

class Int32Decoder : public TypedIterator<const int32_t, Decoder<float>>
{
public:
    Int32Decoder(const int32_t* data)
        : TypedIterator(data) {}

    Int32Decoder()
        : Int32Decoder(nullptr) {}

    float Get() const override
    {
        return static_cast<float>(*m_Iterator);
    }
};

class QASymm8Encoder : public TypedIterator<uint8_t, Encoder<float>>
{
public:
    QASymm8Encoder(uint8_t* data, const float scale, const int32_t offset)
        : TypedIterator(data), m_Scale(scale), m_Offset(offset) {}

    QASymm8Encoder(const float scale, const int32_t offset)
        : QASymm8Encoder(nullptr, scale, offset) {}

    void Set(float right) override
    {
        *m_Iterator = armnn::Quantize<uint8_t>(right, m_Scale, m_Offset);
    }

    float Get() const override
    {
        return armnn::Dequantize(*m_Iterator, m_Scale, m_Offset);
    }

private:
    const float m_Scale;
    const int32_t m_Offset;
};

class QSymm16Encoder : public TypedIterator<int16_t, Encoder<float>>
{
public:
    QSymm16Encoder(int16_t* data, const float scale, const int32_t offset)
        : TypedIterator(data), m_Scale(scale), m_Offset(offset) {}

    QSymm16Encoder(const float scale, const int32_t offset)
        : QSymm16Encoder(nullptr, scale, offset) {}

    void Set(float right) override
    {
        *m_Iterator = armnn::Quantize<int16_t>(right, m_Scale, m_Offset);
    }

    float Get() const override
    {
        return armnn::Dequantize(*m_Iterator, m_Scale, m_Offset);
    }

private:
    const float m_Scale;
    const int32_t m_Offset;
};

class FloatEncoder : public TypedIterator<float, Encoder<float>>
{
public:
    FloatEncoder(float* data)
        : TypedIterator(data) {}

    FloatEncoder()
        : FloatEncoder(nullptr) {}

    void Set(float right) override
    {
        *m_Iterator = right;
    }

    float Get() const override
    {
        return *m_Iterator;
    }
};

class Int32Encoder : public TypedIterator<int32_t, Encoder<float>>
{
public:
    Int32Encoder(int32_t* data)
        : TypedIterator(data) {}

    Int32Encoder()
        : Int32Encoder(nullptr) {}

    void Set(float right) override
    {
        *m_Iterator = static_cast<int32_t>(right);
    }

    float Get() const override
    {
        return static_cast<float>(*m_Iterator);
    }
};

class BooleanEncoder : public TypedIterator<uint8_t, Encoder<bool>>
{
public:
    BooleanEncoder(uint8_t* data)
        : TypedIterator(data) {}

    BooleanEncoder()
        : BooleanEncoder(nullptr) {}

    void Set(bool right) override
    {
        *m_Iterator = right;
    }

    bool Get() const override
    {
        return *m_Iterator;
    }
};

} //namespace armnn
