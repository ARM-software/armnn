//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once
#include <armnn/Types.hpp>

namespace armnnUtils
{

// Provides access to the appropriate indexes for Channels, Height and Width based on DataLayout
class DataLayoutIndexed
{
public:
    DataLayoutIndexed(armnn::DataLayout dataLayout);

    armnn::DataLayout GetDataLayout()    const { return m_DataLayout; }
    unsigned int      GetChannelsIndex() const { return m_ChannelsIndex; }
    unsigned int      GetHeightIndex()   const { return m_HeightIndex; }
    unsigned int      GetWidthIndex()    const { return m_WidthIndex; }

private:
    armnn::DataLayout m_DataLayout;
    unsigned int      m_ChannelsIndex;
    unsigned int      m_HeightIndex;
    unsigned int      m_WidthIndex;
};

// Equality methods
bool operator==(const armnn::DataLayout& dataLayout, const DataLayoutIndexed& indexed);
bool operator==(const DataLayoutIndexed& indexed, const armnn::DataLayout& dataLayout);

} // namespace armnnUtils
