//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <stdio.h>
#include <iterator>

/**
 * Class Array2d is a data structure that represents a two dimensional array.
 * The data is allocated in contiguous memory, arranged row-wise
 * and individual elements can be accessed with the () operator.
 * For example a two dimensional array D of size (M, N) can be accessed:
 *
 *               _|<------------- col size = N  -------->|
 *               |  D(r=0, c=0) D(r=0, c=1)... D(r=0, c=N)
 *               |  D(r=1, c=0) D(r=1, c=1)... D(r=1, c=N)
 *               |  ...
 *    row size = M  ...
 *               |  ...
 *               _  D(r=M, c=0) D(r=M, c=1)... D(r=M, c=N)
 *
 */
template<typename T>
class Array2d
{
private:
    size_t m_rows;
    size_t m_cols;
    T* m_data;

public:
    /**
     * Creates the array2d with the given sizes.
     *
     * @param rows  number of rows.
     * @param cols  number of columns.
     */
    Array2d(unsigned rows, unsigned cols)
    {
        if (rows == 0 || cols == 0) {
            printf("Array2d constructor has 0 size.\n");
            m_data = nullptr;
            return;
        }
        m_rows = rows;
        m_cols = cols;
        m_data = new T[rows * cols];
    }

    ~Array2d()
    {
        delete[] m_data;
    }

    T& operator() (unsigned int row, unsigned int col)
    {
        return m_data[m_cols * row + col];
    }

    T operator() (unsigned int row, unsigned int col) const
    {
        return m_data[m_cols * row + col];
    }

    /**
     * Gets rows number of the current array2d.
     * @return number of rows.
     */
    size_t size(size_t dim)
    {
        switch (dim)
        {
            case 0:
                return m_rows;
            case 1:
                return m_cols;
            default:
                return 0;
        }
    }

    /**
     * Gets the array2d total size.
     */
    size_t totalSize()
    {
        return m_rows * m_cols;
    }

    /**
     * array2d iterator.
     */
    using iterator=T*;
    using const_iterator=T const*;

    iterator begin() { return m_data; }
    iterator end() { return m_data + totalSize(); }
    const_iterator begin() const { return m_data; }
    const_iterator end() const { return m_data + totalSize(); };
};
