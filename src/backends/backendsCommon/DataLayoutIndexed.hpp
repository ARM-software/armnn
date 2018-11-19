//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once
#include <armnn/Types.hpp>

namespace armnn
{

// Provides access to the appropriate indexes for Channels, Height and Width based on DataLayout
class DataLayoutIndexed
{
public:
    DataLayoutIndexed(DataLayout dataLayout) : m_DataLayout(dataLayout)
    {
        switch (dataLayout)
        {
            case DataLayout::NHWC:
                m_ChannelsIndex = 3;
                m_HeightIndex   = 1;
                m_WidthIndex    = 2;
                break;
            case DataLayout::NCHW:
                m_ChannelsIndex = 1;
                m_HeightIndex   = 2;
                m_WidthIndex    = 3;
                break;
            default:
                throw InvalidArgumentException("Unknown DataLayout value: " +
                                               std::to_string(static_cast<int>(dataLayout)));
        }
    }

    DataLayout   GetDataLayout()    const { return m_DataLayout; }
    unsigned int GetChannelsIndex() const { return m_ChannelsIndex; }
    unsigned int GetHeightIndex()   const { return m_HeightIndex; }
    unsigned int GetWidthIndex()    const { return m_WidthIndex; }

private:
    DataLayout   m_DataLayout;
    unsigned int m_ChannelsIndex;
    unsigned int m_HeightIndex;
    unsigned int m_WidthIndex;
};

// Equality methods
bool operator==(const DataLayout& dataLayout, const DataLayoutIndexed& indexed);
bool operator==(const DataLayoutIndexed& indexed, const DataLayout& dataLayout);

}
