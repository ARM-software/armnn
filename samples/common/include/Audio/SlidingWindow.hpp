//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

template<class T>
class SlidingWindow
{
protected:
    T* m_start = nullptr;
    size_t m_dataSize = 0;
    size_t m_size = 0;
    size_t m_stride = 0;
    size_t m_count = 0;
public:

    /**
     * Creates the window slider through the given data.
     *
     * @param data          pointer to the data to slide through.
     * @param dataSize      size in T type elements wise.
     * @param windowSize    sliding window size in T type wise elements.
     * @param stride        stride size in T type wise elements.
     */
    SlidingWindow(T* data, size_t dataSize,
                  size_t windowSize, size_t stride)
    {
        m_start = data;
        m_dataSize = dataSize;
        m_size = windowSize;
        m_stride = stride;
    }

    SlidingWindow() = default;

    ~SlidingWindow() = default;

    /**
     * Get the next data window.
     * @return pointer to the next window, if next window is not available nullptr is returned.
     */
    virtual T* Next()
    {
        if (HasNext())
        {
            m_count++;
            return m_start + Index() * m_stride;
        }
        else
        {
            return nullptr;
        }
    }

    /**
     * Checks if the next data portion is available.
     * @return true if next data portion is available
     */
    bool HasNext()
    {
        return this->m_count < 1 + this->FractionalTotalStrides() && (this->NextWindowStartIndex() < this->m_dataSize);
    }

    /**
     * Resest the slider to the initial position.
     */
    virtual void Reset()
    {
        m_count = 0;
    }

    /**
     * Resest the slider to the initial position.
     */
    virtual size_t GetWindowSize()
    {
        return m_size;
    }

    /**
     * Resets the slider to the start of the new data.
     * New data size MUST be the same as the old one.
     * @param newStart pointer to the new data to slide through.
     */
    virtual void Reset(T* newStart)
    {
        m_start = newStart;
        Reset();
    }

    /**
     * Gets current index of the sliding window.
     * @return current position of the sliding window in number of strides
     */
    size_t Index()
    {
        return m_count == 0? 0: m_count - 1;
    }

    /**
     * Gets the index from the start of the data where the next window will begin.
     * While Index() returns the index of sliding window itself this function returns the index of the data
     * element itself.
     * @return Index from the start of the data where the next sliding window will begin.
     */
    virtual size_t NextWindowStartIndex()
    {
        return m_count == 0? 0: ((m_count) * m_stride);
    }

    /**
     * Go to given sliding window index.
     * @param index new position of the sliding window. if index is invalid (greater than possible range of strides)
     *              then next call to Next() will return nullptr.
     */
    void FastForward(size_t index)
    {
        m_count = index;
    }

    /**
     * Calculates whole number of times the window can stride through the given data.
     * @return maximum number of strides.
     */
    size_t TotalStrides()
    {
        if (m_size > m_dataSize)
        {
            return 0;
        }
        return ((m_dataSize - m_size)/m_stride);
    }

    /**
     * Calculates number of times the window can stride through the given data. May not be a whole number.
     * @return Number of strides to cover all data.
     */
    float FractionalTotalStrides()
    {
        if(this->m_size > this->m_dataSize)
        {
            return this->m_dataSize / this->m_size;
        }
        else
        {
            return ((this->m_dataSize - this->m_size)/ static_cast<float>(this->m_stride));
        }

    }

    /**
     * Calculates the remaining data left to be processed
     * @return The remaining unprocessed data
     */
    int RemainingData()
    {
        return this->m_dataSize - this->NextWindowStartIndex();
    }
};