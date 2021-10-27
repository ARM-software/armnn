//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

size_t GetMinPossibleMemorySize(const std::vector<armnn::MemBlock>& blocks)
{
    unsigned int maxLifetime = 0;
    for (auto& block: blocks)
    {
        maxLifetime = std::max(maxLifetime, block.m_EndOfLife);
    }
    maxLifetime++;

    std::vector<size_t> lifetimes(maxLifetime);
    for (const auto& block : blocks)
    {
        for (auto lifetime = block.m_StartOfLife; lifetime <= block.m_EndOfLife; ++lifetime)
        {
            lifetimes[lifetime] += block.m_MemSize;
        }
    }
    return *std::max_element(lifetimes.begin(), lifetimes.end());
}

// Generated from fsrcnn_720p.tflite
std::vector<armnn::MemBlock> fsrcnn
{
        { 0, 1, 691200, 0, 0 },
        { 1, 3, 7372800, 0, 1 },
        { 2, 5, 7372800, 0, 2 },
        { 3, 7, 1843200, 0, 3 },
        { 4, 9, 1843200, 0, 4 },
        { 5, 11, 1843200, 0, 5 },
        { 6, 13, 1843200, 0, 6 },
        { 7, 15, 1843200, 0, 7 },
        { 8, 17, 1843200, 0, 8 },
        { 9, 19, 7372800, 0, 9 },
        { 10, 21, 7372800, 0, 10 },
        { 11, 23, 2764800, 0, 11 },
        { 12, 25, 2764800, 0, 12 },
        { 13, 27, 2764800, 0, 13 }
};