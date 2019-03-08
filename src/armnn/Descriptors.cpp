//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include "armnn/Descriptors.hpp"

#include <algorithm>
#include <array>
#include <vector>

#include <boost/format.hpp>
#include <boost/log/trivial.hpp>
#include <boost/numeric/conversion/cast.hpp>

namespace armnn
{

PermutationVector::PermutationVector(const ValueType *dimMappings, const SizeType numDimMappings)
{
    // Validation

    if (numDimMappings > MaxNumOfTensorDimensions)
    {
        boost::format fmt("The number of mappings (%1%) cannot be greater "
                          "than the maximum number of dimensions supported (%2%)");
        throw InvalidArgumentException(boost::str(fmt % numDimMappings % MaxNumOfTensorDimensions));
    }

    if ((dimMappings == nullptr) && (numDimMappings != 0))
    {
        throw InvalidArgumentException("Dimension mappings must not be NULL if the number of mappings is positive");
    }

    for (SizeType i = 0; i < numDimMappings; ++i)
    {
        const ValueType dstIndex = dimMappings[i];
        if (dstIndex >= numDimMappings)
        {
            boost::format fmt("Dimension mapping at index %1% is invalid: %2% is outside of the valid range [0,%3%]");
            throw InvalidArgumentException(boost::str(fmt % i % dstIndex % (numDimMappings - 1)));
        }
    }

    // Validation: Detect duplicates
    {
        std::array<bool, MaxNumOfTensorDimensions> observedDims;
        observedDims.fill(false);

        for (SizeType i = 0; i < numDimMappings; ++i)
        {
            const ValueType dstIndex = dimMappings[i];
            if (observedDims[dstIndex])
            {
                throw InvalidArgumentException("Invalid dimension mappings: Two or more source dimensions are mapped "
                    "to the same output dimension");
            }
            observedDims[dstIndex] = true;
        }
    }

    // Initialize
    for (SizeType i = 0; i < numDimMappings; ++i)
    {
        m_DimMappings[i] = dimMappings[i];
    }
    m_NumDimMappings = numDimMappings;
}

PermutationVector::PermutationVector(std::initializer_list<ValueType> dimMappings)
    : PermutationVector(dimMappings.begin(), boost::numeric_cast<SizeType>(dimMappings.size()))
{
}

OriginsDescriptor::OriginsDescriptor()
: m_ConcatAxis(1)
, m_NumViews(0)
, m_NumDimensions(0)
, m_ViewOrigins(nullptr)
{}

OriginsDescriptor::OriginsDescriptor(uint32_t numViews, uint32_t numDimensions /*= 4*/)
: m_ConcatAxis(1)
, m_NumViews(numViews)
, m_NumDimensions(numDimensions)
, m_ViewOrigins(numViews && numDimensions > 0 ? new uint32_t *[numViews]() : nullptr)
{
    for (uint32_t i = 0; m_NumDimensions > 0 && i < m_NumViews; ++i)
    {
        m_ViewOrigins[i] = new uint32_t[m_NumDimensions]();
    }
}

OriginsDescriptor::OriginsDescriptor(const OriginsDescriptor& other)
: m_ConcatAxis(other.m_ConcatAxis)
, m_NumViews(other.m_NumViews)
, m_NumDimensions(other.m_NumDimensions)
, m_ViewOrigins(other.m_NumViews && other.m_NumDimensions > 0 ? new uint32_t *[other.m_NumViews]() : nullptr)
{
    for (uint32_t i = 0; m_NumDimensions > 0 && i < m_NumViews; ++i)
    {
        m_ViewOrigins[i] = new uint32_t[m_NumDimensions]();
        memcpy(m_ViewOrigins[i], other.m_ViewOrigins[i], m_NumDimensions * sizeof(uint32_t));
    }
}

OriginsDescriptor::OriginsDescriptor(OriginsDescriptor&& other)
: OriginsDescriptor()
{
    swap(*this, other);
}

OriginsDescriptor::~OriginsDescriptor()
{
    for (uint32_t i = 0; m_NumDimensions > 0 && i < m_NumViews; ++i)
    {
        delete[] m_ViewOrigins[i];
    }
    delete[] m_ViewOrigins;
}

OriginsDescriptor& OriginsDescriptor::operator=(OriginsDescriptor rhs)
{
    swap(*this, rhs);
    return *this;
}

void OriginsDescriptor::SetConcatAxis(unsigned int concatAxis)
{
    m_ConcatAxis = concatAxis;
}
unsigned int OriginsDescriptor::GetConcatAxis() const
{
    return m_ConcatAxis;
}

Status OriginsDescriptor::SetViewOriginCoord(uint32_t view, uint32_t coord, uint32_t value)
{
    if (view >= m_NumViews)
    {
        BOOST_LOG_TRIVIAL(error) << "OriginsDescriptor::SetViewOriginCoord: view argument:" << view <<
            " is out of range";
        return Status::Failure;
    }
    if (coord >= m_NumDimensions)
    {
        BOOST_LOG_TRIVIAL(error) << "OriginsDescriptor::SetViewOriginCoord: coord argument:" << coord <<
            " is out of range";
        return Status::Failure;
    }

    m_ViewOrigins[view][coord] = value;
    return Status::Success;
}


uint32_t OriginsDescriptor::GetNumViews() const
{
    return m_NumViews;
}

uint32_t OriginsDescriptor::GetNumDimensions() const
{
    return m_NumDimensions;
}

const uint32_t* OriginsDescriptor::GetViewOrigin(uint32_t idx) const
{
    return m_ViewOrigins ? m_ViewOrigins[idx] : nullptr;
}


// Reorders the viewOrigins in accordance with the indices presented in newOrdering array.
void OriginsDescriptor::ReorderOrigins(unsigned int*  newOrdering, unsigned int numNewOrdering)
{
    BOOST_ASSERT_MSG(m_NumViews == numNewOrdering, "number of views must match number of "
        "elements in the new ordering array");
    std::vector<uint32_t*> viewOrigins(&m_ViewOrigins[0], &m_ViewOrigins[m_NumViews]);

    for (unsigned int i = 0; i < numNewOrdering; ++i)
    {
        m_ViewOrigins[i] = viewOrigins[newOrdering[i]];
    }
}

ViewsDescriptor::ViewsDescriptor()
: m_Origins()
, m_ViewSizes(nullptr)
{}

ViewsDescriptor::ViewsDescriptor(uint32_t numViews, uint32_t numDimensions /*= 4*/)
    : m_Origins(numViews, numDimensions)
    , m_ViewSizes(numViews > 0 && numDimensions > 0 ?
                      new uint32_t *[numViews]() : nullptr)
{
    if (m_ViewSizes)
    {
        for (uint32_t i = 0; GetNumDimensions() > 0 && i < GetNumViews(); ++i)
        {
            m_ViewSizes[i] = new uint32_t[GetNumDimensions()]();
        }
    }
}

ViewsDescriptor::ViewsDescriptor(const ViewsDescriptor& other)
    : m_Origins(other.m_Origins)
    , m_ViewSizes(other.GetNumViews() > 0 && other.GetNumDimensions() > 0 ?
                      new uint32_t *[other.GetNumViews()]() : nullptr)
{
    if (m_ViewSizes)
    {
        for (uint32_t i = 0; GetNumDimensions() > 0 && i < GetNumViews(); ++i)
        {
            m_ViewSizes[i] = new uint32_t[GetNumDimensions()]();
            memcpy(m_ViewSizes[i], other.m_ViewSizes[i], GetNumDimensions() * sizeof(uint32_t));
        }
    }
}

ViewsDescriptor::ViewsDescriptor(ViewsDescriptor&& other)
    : ViewsDescriptor()
{
    swap(*this, other);
}

ViewsDescriptor::~ViewsDescriptor()
{
    if (m_ViewSizes)
    {
        for (uint32_t i = 0; GetNumDimensions() > 0 && i < GetNumViews(); ++i)
        {
            delete[] m_ViewSizes[i];
        }
        delete[] m_ViewSizes;
    }
}

ViewsDescriptor& ViewsDescriptor::operator=(ViewsDescriptor rhs)
{
    swap(*this, rhs);
    return *this;
}

uint32_t ViewsDescriptor::GetNumViews() const
{
    return m_Origins.GetNumViews();
}

uint32_t ViewsDescriptor::GetNumDimensions() const
{
    return m_Origins.GetNumDimensions();
}

const uint32_t* ViewsDescriptor::GetViewOrigin(uint32_t idx) const
{
    return m_Origins.GetViewOrigin(idx);
}

Status ViewsDescriptor::SetViewOriginCoord(uint32_t view, uint32_t coord, uint32_t value)
{
    return m_Origins.SetViewOriginCoord(view, coord, value);
}

Status ViewsDescriptor::SetViewSize(uint32_t view, uint32_t coord, uint32_t value)
{
    if (!m_ViewSizes)
    {
        BOOST_LOG_TRIVIAL(error) << "ViewsDescriptor::SetViewSize: invalid view sizes";
        return Status::Failure;
    }

    if (view >= GetNumViews())
    {
        BOOST_LOG_TRIVIAL(error) << "ViewsDescriptor::SetViewSize: view argument:" << view <<
                                 " is out of range";
        return Status::Failure;
    }
    if (coord >= GetNumDimensions())
    {
        BOOST_LOG_TRIVIAL(error) << "ViewsDescriptor::SetViewSize: coord argument:" << coord <<
                                 " is out of range";
        return Status::Failure;
    }

    m_ViewSizes[view][coord] = value;
    return Status::Success;
}

const uint32_t* ViewsDescriptor::GetViewSizes(uint32_t idx) const
{
    return m_ViewSizes ? m_ViewSizes[idx] : nullptr;
}

const OriginsDescriptor& ViewsDescriptor::GetOrigins() const
{
    return m_Origins;
}

void swap(OriginsDescriptor& first, OriginsDescriptor& second)
{
    using std::swap;
    swap(first.m_NumViews, second.m_NumViews);
    swap(first.m_NumDimensions, second.m_NumDimensions);
    swap(first.m_ViewOrigins, second.m_ViewOrigins);
    swap(first.m_ConcatAxis, second.m_ConcatAxis);
}

void swap(ViewsDescriptor& first, ViewsDescriptor& second)
{
    using std::swap;
    swap(first.m_Origins, second.m_Origins);
    swap(first.m_ViewSizes, second.m_ViewSizes);
}

int StridedSliceDescriptor::GetStartForAxis(const TensorShape& inputShape,
                                            unsigned int axis) const
{
    int start = m_Begin[axis];

    if (m_BeginMask & (1 << axis))
    {
        if (m_Stride[axis] > 0)
        {
            start = std::numeric_limits<int>::min();
        }
        else
        {
            start = std::numeric_limits<int>::max();
        }
    }

    const int axisSize = boost::numeric_cast<int>(inputShape[axis]);
    if (start < 0)
    {
        start += (axisSize);
    }

    return std::max(0, std::min(start, axisSize - 1));

}

int StridedSliceDescriptor::GetStopForAxis(const TensorShape& inputShape,
                                           unsigned int axis,
                                           int startForAxis) const
{

    if (m_ShrinkAxisMask & (1 << axis))
    {
        return startForAxis + 1;
    }

    int stop = m_End[axis];

    if (m_EndMask & (1 << axis))
    {
        if (m_Stride[axis] > 0)
        {
            stop = std::numeric_limits<int>::max();
        }
        else
        {
            stop = std::numeric_limits<int>::min();
        }
    }

    const int axisSize = boost::numeric_cast<int>(inputShape[axis]);
    if (stop < 0)
    {
        stop += axisSize;
    }

    return m_Stride[axis] > 0 ? std::max(0, std::min(stop, axisSize)) :
                                std::max(-1, std::min(stop, axisSize - 1));

}

}
