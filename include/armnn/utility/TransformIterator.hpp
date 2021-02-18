//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <iterator>

namespace armnn
{

template<typename Function,
        typename Iterator,
        typename Category = typename std::iterator_traits<Iterator>::iterator_category,
        typename T = typename std::iterator_traits<Iterator>::value_type,
        typename Distance = typename std::iterator_traits<Iterator>::difference_type,
        typename Pointer = typename std::iterator_traits<Iterator>::pointer,
        typename Reference =
        typename std::result_of<const Function(typename std::iterator_traits<Iterator>::reference)>::type
>
class TransformIterator : public std::iterator<Category, T, Distance, Pointer, Reference>
{

public:

    TransformIterator() = default;
    TransformIterator(TransformIterator const& transformIterator) = default;
    TransformIterator(TransformIterator&& transformIterator) = default;

    TransformIterator(Iterator& it, Function fn) : m_it(it), m_fn(fn) {}
    TransformIterator(Iterator&& it, Function fn) : m_it(it), m_fn(fn) {}

    ~TransformIterator() = default;

    TransformIterator operator=(TransformIterator const& transformIterator)
    {
        return { transformIterator.m_it, transformIterator.m_fn };
    }

    TransformIterator operator=(TransformIterator&& transformIterator)
    {
        return { transformIterator.m_it, transformIterator.m_fn };
    }

    TransformIterator operator++() {++m_it; return *this;}
    TransformIterator operator--() {--m_it; return *this;}

    TransformIterator operator++() const {++m_it; return *this;}
    TransformIterator operator--() const {--m_it; return *this;}

    TransformIterator operator++(int n) const {m_it += n; return *this;}
    TransformIterator operator--(int n) const {m_it -= n; return *this;}

    TransformIterator operator[](Distance n) const {m_it[n]; return  *this;};

    Distance operator-(const TransformIterator& other) {return m_it - other.m_it;}

    TransformIterator operator-(const Distance n) {return {m_it - n, m_fn};}
    TransformIterator operator+(const Distance n) {return {m_it + n, m_fn};}

    bool operator>(const TransformIterator& rhs) const {return m_it > rhs.m_it;}
    bool operator<(const TransformIterator& rhs) const {return m_it < rhs.m_it;}
    bool operator>=(const TransformIterator& rhs) const {return m_it >= rhs.m_it;}
    bool operator<=(const TransformIterator& rhs) const {return m_it <= rhs.m_it;}

    bool operator==(TransformIterator other) const {return (m_it == other.m_it);}
    bool operator!=(TransformIterator other) const {return !(m_it == other.m_it);}

    Reference operator*() const {return m_fn(*m_it);}

private:
    Iterator m_it;
    const Function m_fn;
};

template<typename Function, typename Iterator>
constexpr TransformIterator<Function, Iterator> MakeTransformIterator(Iterator i, Function f)
{
    return TransformIterator<Function, Iterator>(i, f);
}

}