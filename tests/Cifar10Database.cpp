//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include "Cifar10Database.hpp"

#include <armnn/Logging.hpp>
#include <armnn/utility/NumericCast.hpp>

#include <fstream>
#include <vector>

constexpr unsigned int g_kCifar10ImageByteSize = 1 + 3 * 32 * 32;

Cifar10Database::Cifar10Database(const std::string& binaryFileDirectory, bool rgbPack)
    : m_BinaryDirectory(binaryFileDirectory), m_RgbPack(rgbPack)
{
}

std::unique_ptr<Cifar10Database::TTestCaseData> Cifar10Database::GetTestCaseData(unsigned int testCaseId)
{
    std::vector<unsigned char> I(g_kCifar10ImageByteSize);

    std::string fullpath = m_BinaryDirectory + std::string("test_batch.bin");

    std::ifstream fileStream(fullpath, std::ios::binary);
    if (!fileStream.is_open())
    {
        ARMNN_LOG(fatal) << "Failed to load " << fullpath;
        return nullptr;
    }

    fileStream.seekg(testCaseId * g_kCifar10ImageByteSize, std::ios_base::beg);
    fileStream.read(reinterpret_cast<char*>(&I[0]), g_kCifar10ImageByteSize);

    if (!fileStream.good())
    {
        ARMNN_LOG(fatal) << "Failed to read " << fullpath;
        return nullptr;
    }


    std::vector<float> inputImageData;
    inputImageData.resize(g_kCifar10ImageByteSize - 1);

    unsigned int step;
    unsigned int countR_o;
    unsigned int countG_o;
    unsigned int countB_o;
    unsigned int countR = 1;
    unsigned int countG = 1 + 32 * 32;
    unsigned int countB = 1 + 2 * 32 * 32;

    if (m_RgbPack)
    {
        countR_o = 0;
        countG_o = 1;
        countB_o = 2;
        step = 3;
    }
    else
    {
        countR_o = 0;
        countG_o = 32 * 32;
        countB_o = 2 * 32 * 32;
        step = 1;
    }

    for (unsigned int h = 0; h < 32; h++)
    {
        for (unsigned int w = 0; w < 32; w++)
        {
            // Static_cast of unsigned char is safe with float
            inputImageData[countR_o] = static_cast<float>(I[countR++]);
            inputImageData[countG_o] = static_cast<float>(I[countG++]);
            inputImageData[countB_o] = static_cast<float>(I[countB++]);

            countR_o += step;
            countG_o += step;
            countB_o += step;
        }
    }

    const unsigned int label = armnn::numeric_cast<unsigned int>(I[0]);
    return std::make_unique<TTestCaseData>(label, std::move(inputImageData));
}
