//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <armnnUtils/DataLayoutIndexed.hpp>

using namespace armnn;

namespace armnnUtils
{

DataLayoutIndexed::DataLayoutIndexed(armnn::DataLayout dataLayout)
    : m_DataLayout(dataLayout)
{
    switch (dataLayout)
    {
        case armnn::DataLayout::NHWC:
            m_ChannelsIndex = 3;
            m_HeightIndex   = 1;
            m_WidthIndex    = 2;
            break;
        case armnn::DataLayout::NCHW:
            m_ChannelsIndex = 1;
            m_HeightIndex   = 2;
            m_WidthIndex    = 3;
            break;
        case armnn::DataLayout::NDHWC:
            m_DepthIndex    = 1;
            m_HeightIndex   = 2;
            m_WidthIndex    = 3;
            m_ChannelsIndex = 4;
            break;
        case armnn::DataLayout::NCDHW:
            m_ChannelsIndex = 1;
            m_DepthIndex    = 2;
            m_HeightIndex   = 3;
            m_WidthIndex    = 4;
            break;
        default:
            throw armnn::InvalidArgumentException("Unknown DataLayout value: " +
                                                  std::to_string(static_cast<int>(dataLayout)));
    }
}

bool operator==(const DataLayout& dataLayout, const DataLayoutIndexed& indexed)
{
    return dataLayout == indexed.GetDataLayout();
}

bool operator==(const DataLayoutIndexed& indexed, const DataLayout& dataLayout)
{
    return indexed.GetDataLayout() == dataLayout;
}

} // namespace armnnUtils
