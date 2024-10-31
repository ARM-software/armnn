//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include "MnistDatabase.hpp"

#include <armnn/Logging.hpp>

#include <fstream>
#include <vector>

constexpr int g_kMnistImageByteSize = 28 * 28;

void EndianSwap(unsigned int &x)
{
    x = (x >> 24) | ((x << 8) & 0x00FF0000) | ((x >> 8) & 0x0000FF00) | (x << 24);
}

MnistDatabase::MnistDatabase(const std::string& binaryFileDirectory, bool scaleValues)
    : m_BinaryDirectory(binaryFileDirectory)
    , m_ScaleValues(scaleValues)
{
}

std::unique_ptr<MnistDatabase::TTestCaseData> MnistDatabase::GetTestCaseData(unsigned int testCaseId)
{
    std::vector<unsigned char> I(g_kMnistImageByteSize);
    unsigned int label = 0;

    std::string imagePath = m_BinaryDirectory + std::string("t10k-images.idx3-ubyte");
    std::string labelPath = m_BinaryDirectory + std::string("t10k-labels.idx1-ubyte");

    std::ifstream imageStream(imagePath, std::ios::binary);
    std::ifstream labelStream(labelPath, std::ios::binary);

    if (!imageStream.is_open())
    {
        ARMNN_LOG(fatal) << "Failed to load " << imagePath;
        return nullptr;
    }
    if (!labelStream.is_open())
    {
        ARMNN_LOG(fatal) << "Failed to load " << imagePath;
        return nullptr;
    }

    unsigned int magic, num, row, col;

    // Checks the files have the correct header.
    imageStream.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    if (magic != 0x03080000)
    {
        ARMNN_LOG(fatal) << "Failed to read " << imagePath;
        return nullptr;
    }
    labelStream.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    if (magic != 0x01080000)
    {
        ARMNN_LOG(fatal) << "Failed to read " << labelPath;
        return nullptr;
    }

    // Endian swaps the image and label file - all the integers in the files are stored in MSB first(high endian)
    // format, hence it needs to flip the bytes of the header if using it on Intel processors or low-endian machines
    labelStream.read(reinterpret_cast<char*>(&num), sizeof(num));
    imageStream.read(reinterpret_cast<char*>(&num), sizeof(num));
    EndianSwap(num);
    imageStream.read(reinterpret_cast<char*>(&row), sizeof(row));
    EndianSwap(row);
    imageStream.read(reinterpret_cast<char*>(&col), sizeof(col));
    EndianSwap(col);

    // Reads image and label into memory.
    imageStream.seekg(testCaseId * g_kMnistImageByteSize, std::ios_base::cur);
    imageStream.read(reinterpret_cast<char*>(&I[0]), g_kMnistImageByteSize);
    labelStream.seekg(testCaseId, std::ios_base::cur);
    labelStream.read(reinterpret_cast<char*>(&label), 1);

    if (!imageStream.good())
    {
        ARMNN_LOG(fatal) << "Failed to read " << imagePath;
        return nullptr;
    }
    if (!labelStream.good())
    {
        ARMNN_LOG(fatal) << "Failed to read " << labelPath;
        return nullptr;
    }

    std::vector<float> inputImageData;
    inputImageData.resize(g_kMnistImageByteSize);

    for (unsigned int i = 0; i < col * row; ++i)
    {
        // Static_cast of unsigned char is safe with float
        inputImageData[i] = static_cast<float>(I[i]);

        if(m_ScaleValues)
        {
            inputImageData[i] /= 255.0f;
        }
    }

    return std::make_unique<TTestCaseData>(label, std::move(inputImageData));
}
