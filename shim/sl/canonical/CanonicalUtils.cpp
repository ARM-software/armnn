//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#define LOG_TAG "arm-armnn-sl"

#include "CanonicalUtils.hpp"

#include <armnn/Utils.hpp>
#include <armnn/utility/Assert.hpp>
#include <armnnSerializer/ISerializer.hpp>
#include <armnnUtils/Permute.hpp>

#include <ghc/filesystem.hpp>
namespace fs = ghc::filesystem;
#include <half/half.hpp>
#include <log/log.h>

#include <cassert>
#include <cerrno>
#include <cinttypes>
#include <cstdio>
#include <sstream>
#include <time.h>
#include <variant>

namespace armnn
{
using Half = half_float::half; //import half float implementation
} //namespace armnn

using namespace android;
using namespace android::nn;

namespace armnn_driver
{
const armnn::PermutationVector g_DontPermute{};

void SwizzleAndroidNn4dTensorToArmNn(armnn::TensorInfo& tensorInfo,
                                     const void* input,
                                     void* output,
                                     const armnn::PermutationVector& mappings)
{
    assert(tensorInfo.GetNumDimensions() == 4U);

    armnn::DataType dataType = tensorInfo.GetDataType();
    switch (dataType)
    {
    case armnn::DataType::Float16:
    case armnn::DataType::Float32:
    case armnn::DataType::QAsymmU8:
    case armnn::DataType::QSymmS8:
    case armnn::DataType::QAsymmS8:
        // First swizzle tensor info
        tensorInfo = armnnUtils::Permuted(tensorInfo, mappings);
        // Then swizzle tensor data
        armnnUtils::Permute(tensorInfo.GetShape(), mappings, input, output, armnn::GetDataTypeSize(dataType));
        break;
    default:
        VLOG(DRIVER) << "Unknown armnn::DataType for swizzling";
        assert(0);
    }
}

void* GetMemoryFromPool(DataLocation location, const std::vector<android::nn::RunTimePoolInfo>& memPools)
{
    // find the location within the pool
    assert(location.poolIndex < memPools.size());

    const android::nn::RunTimePoolInfo& memPool = memPools[location.poolIndex];
    uint8_t* memPoolBuffer = memPool.getBuffer();
    uint8_t* memory = memPoolBuffer + location.offset;
    return memory;
}

void* GetMemoryFromPointer(const Request::Argument& requestArg)
{
    // get the pointer memory
    auto ptrMemory = std::visit([](std::variant<const void*, void*>&& memory)
                                {
                                    if (std::holds_alternative<const void*>(memory))
                                    {
                                        auto ptr = std::get<const void*>(memory);
                                        auto ptrMemory = static_cast<const uint8_t*>(ptr);
                                        return const_cast<uint8_t*>(ptrMemory);
                                    }
                                    else
                                    {
                                        auto ptr = std::get<void*>(memory);
                                        return static_cast<uint8_t*>(ptr);
                                    }
                                }, requestArg.location.pointer);
    return ptrMemory;
}

armnn::TensorInfo GetTensorInfoForOperand(const Operand& operand)
{
    using namespace armnn;
    bool perChannel = false;
    bool isScalar   = false;

    DataType type;
    switch (operand.type)
    {
        case OperandType::TENSOR_BOOL8:
            type = armnn::DataType::Boolean;
            break;
        case OperandType::TENSOR_FLOAT32:
            type = armnn::DataType::Float32;
            break;
        case OperandType::TENSOR_FLOAT16:
            type = armnn::DataType::Float16;
            break;
        case OperandType::TENSOR_QUANT8_ASYMM:
            type = armnn::DataType::QAsymmU8;
            break;
        case OperandType::TENSOR_QUANT8_SYMM_PER_CHANNEL:
            perChannel=true;
            ARMNN_FALLTHROUGH;
        case OperandType::TENSOR_QUANT8_SYMM:
            type = armnn::DataType::QSymmS8;
            break;
        case OperandType::TENSOR_QUANT16_SYMM:
            type = armnn::DataType::QSymmS16;
            break;
        case OperandType::TENSOR_INT32:
            type = armnn::DataType::Signed32;
            break;
        case OperandType::INT32:
            type = armnn::DataType::Signed32;
            isScalar = true;
            break;
        case OperandType::TENSOR_QUANT8_ASYMM_SIGNED:
            type = armnn::DataType::QAsymmS8;
            break;
        default:
            throw UnsupportedOperand<OperandType>(operand.type);
    }

    TensorInfo ret;
    if (isScalar)
    {
        ret = TensorInfo(TensorShape(armnn::Dimensionality::Scalar), type);
    }
    else
    {
        if (operand.dimensions.size() == 0)
        {
            TensorShape tensorShape(Dimensionality::NotSpecified);
            ret = TensorInfo(tensorShape, type);
        }
        else
        {
            bool dimensionsSpecificity[5] = { true, true, true, true, true };
            int count = 0;
            std::for_each(operand.dimensions.data(),
                          operand.dimensions.data() +  operand.dimensions.size(),
                          [&](const unsigned int val)
                          {
                              if (val == 0)
                              {
                                  dimensionsSpecificity[count] = false;
                              }
                              count++;
                          });

            TensorShape tensorShape(operand.dimensions.size(), operand.dimensions.data(), dimensionsSpecificity);
            ret = TensorInfo(tensorShape, type);
        }
    }

    if (perChannel)
    {
        // ExtraParams is expected to be of type channelQuant
        const auto& perAxisQuantParams = std::get<Operand::SymmPerChannelQuantParams>(operand.extraParams);

        ret.SetQuantizationScales(perAxisQuantParams.scales);
        ret.SetQuantizationDim(MakeOptional<unsigned int>(perAxisQuantParams.channelDim));
    }
    else
    {
        ret.SetQuantizationScale(operand.scale);
        ret.SetQuantizationOffset(operand.zeroPoint);
    }
    return ret;
}

std::string GetOperandSummary(const Operand& operand)
{
    std::stringstream ss;
    ss << "operand dimensions: [ ";
    for (unsigned int i = 0; i < operand.dimensions.size(); ++i)
    {
        ss << operand.dimensions[i] << " ";
    }
    ss << "] operand type: " << operand.type;
    return ss.str();
}

template <typename TensorType>
using DumpElementFunction = void (*)(const TensorType& tensor,
                                     unsigned int elementIndex,
                                     std::ofstream& fileStream);

namespace
{
template <typename TensorType, typename ElementType, typename PrintableType = ElementType>
void DumpTensorElement(const TensorType& tensor, unsigned int elementIndex, std::ofstream& fileStream)
{
    const ElementType* elements = reinterpret_cast<const ElementType*>(tensor.GetMemoryArea());
    fileStream << static_cast<PrintableType>(elements[elementIndex]) << " ";
}

} // namespace
template <typename TensorType>
void DumpTensor(const std::string& dumpDir,
                const std::string& requestName,
                const std::string& tensorName,
                const TensorType& tensor)
{
    // The dump directory must exist in advance.
    fs::path dumpPath = dumpDir;
    const fs::path fileName = dumpPath / (requestName + "_" + tensorName + ".dump");

    std::ofstream fileStream;
    fileStream.open(fileName.c_str(), std::ofstream::out | std::ofstream::trunc);

    if (!fileStream.good())
    {
        VLOG(DRIVER) << "Could not open file %s for writing" << fileName.c_str();
        return;
    }

    DumpElementFunction<TensorType> dumpElementFunction = nullptr;

    switch (tensor.GetDataType())
    {
        case armnn::DataType::Float32:
        {
            dumpElementFunction = &DumpTensorElement<TensorType, float>;
            break;
        }
        case armnn::DataType::QAsymmU8:
        {
            dumpElementFunction = &DumpTensorElement<TensorType, uint8_t, uint32_t>;
            break;
        }
        case armnn::DataType::Signed32:
        {
            dumpElementFunction = &DumpTensorElement<TensorType, int32_t>;
            break;
        }
        case armnn::DataType::Float16:
        {
            dumpElementFunction = &DumpTensorElement<TensorType, armnn::Half>;
            break;
        }
        case armnn::DataType::QAsymmS8:
        {
            dumpElementFunction = &DumpTensorElement<TensorType, int8_t, int32_t>;
            break;
        }
        case armnn::DataType::Boolean:
        {
            dumpElementFunction = &DumpTensorElement<TensorType, bool>;
            break;
        }
        default:
        {
            dumpElementFunction = nullptr;
        }
    }

    if (dumpElementFunction != nullptr)
    {
        const unsigned int numDimensions = tensor.GetNumDimensions();
        const armnn::TensorShape shape = tensor.GetShape();

        if (!shape.AreAllDimensionsSpecified())
        {
            fileStream << "Cannot dump tensor elements: not all dimensions are specified" << std::endl;
            return;
        }
        fileStream << "# Number of elements " << tensor.GetNumElements() << std::endl;

        if (numDimensions == 0)
        {
            fileStream << "# Shape []" << std::endl;
            return;
        }
        fileStream << "# Shape [" << shape[0];
        for (unsigned int d = 1; d < numDimensions; ++d)
        {
            fileStream << "," << shape[d];
        }
        fileStream << "]" << std::endl;
        fileStream << "Each line contains the data of each of the elements of dimension0. In NCHW and NHWC, each line"
                      " will be a batch" << std::endl << std::endl;

        // Split will create a new line after all elements of the first dimension
        // (in a 4, 3, 2, 3 tensor, there will be 4 lines of 18 elements)
        unsigned int split = 1;
        if (numDimensions == 1)
        {
            split = shape[0];
        }
        else
        {
            for (unsigned int i = 1; i < numDimensions; ++i)
            {
                split *= shape[i];
            }
        }

        // Print all elements in the tensor
        for (unsigned int elementIndex = 0; elementIndex < tensor.GetNumElements(); ++elementIndex)
        {
            (*dumpElementFunction)(tensor, elementIndex, fileStream);

            if ( (elementIndex + 1) % split == 0 )
            {
                fileStream << std::endl;
            }
        }
        fileStream << std::endl;
    }
    else
    {
        fileStream << "Cannot dump tensor elements: Unsupported data type "
            << static_cast<unsigned int>(tensor.GetDataType()) << std::endl;
    }

    if (!fileStream.good())
    {
        VLOG(DRIVER) << "An error occurred when writing to file %s" << fileName.c_str();
    }
}

template void DumpTensor<armnn::ConstTensor>(const std::string& dumpDir,
                                             const std::string& requestName,
                                             const std::string& tensorName,
                                             const armnn::ConstTensor& tensor);

template void DumpTensor<armnn::Tensor>(const std::string& dumpDir,
                                        const std::string& requestName,
                                        const std::string& tensorName,
                                        const armnn::Tensor& tensor);

void DumpJsonProfilingIfRequired(bool gpuProfilingEnabled,
                                 const std::string& dumpDir,
                                 armnn::NetworkId networkId,
                                 const armnn::IProfiler* profiler)
{
    // Check if profiling is required.
    if (!gpuProfilingEnabled)
    {
        return;
    }

    // The dump directory must exist in advance.
    if (dumpDir.empty())
    {
        return;
    }

    ARMNN_ASSERT(profiler);

    // Set the name of the output profiling file.
    fs::path dumpPath = dumpDir;
    const fs::path fileName = dumpPath / (std::to_string(networkId) + "_profiling.json");

    // Open the ouput file for writing.
    std::ofstream fileStream;
    fileStream.open(fileName.c_str(), std::ofstream::out | std::ofstream::trunc);

    if (!fileStream.good())
    {
        VLOG(DRIVER) << "Could not open file %s for writing" << fileName.c_str();
        return;
    }

    // Write the profiling info to a JSON file.
    profiler->Print(fileStream);
}

std::string ExportNetworkGraphToDotFile(const armnn::IOptimizedNetwork& optimizedNetwork,
                                        const std::string& dumpDir)
{
    std::string fileName;
    // The dump directory must exist in advance.
    if (dumpDir.empty())
    {
        return fileName;
    }

    std::string timestamp = GetFileTimestamp();
    if (timestamp.empty())
    {
        return fileName;
    }

    // Set the name of the output .dot file.
    fs::path dumpPath = dumpDir;
    fs::path tempFilePath = dumpPath / (timestamp + "_networkgraph.dot");
    fileName = tempFilePath.string();

    VLOG(DRIVER) << "Exporting the optimized network graph to file: %s" << fileName.c_str();

    // Write the network graph to a dot file.
    std::ofstream fileStream;
    fileStream.open(fileName, std::ofstream::out | std::ofstream::trunc);

    if (!fileStream.good())
    {
        VLOG(DRIVER) << "Could not open file %s for writing" << fileName.c_str();
        return fileName;
    }

    if (optimizedNetwork.SerializeToDot(fileStream) != armnn::Status::Success)
    {
        VLOG(DRIVER) << "An error occurred when writing to file %s" << fileName.c_str();
    }
    return fileName;
}

std::string SerializeNetwork(const armnn::INetwork& network,
                             const std::string& dumpDir,
                             std::vector<uint8_t>& dataCacheData,
                             bool dataCachingActive)
{
    std::string fileName;
    bool bSerializeToFile = true;
    if (dumpDir.empty())
    {
        bSerializeToFile = false;
    }
    else
    {
        std::string timestamp = GetFileTimestamp();
        if (timestamp.empty())
        {
            bSerializeToFile = false;
        }
    }
    if (!bSerializeToFile && !dataCachingActive)
    {
        return fileName;
    }

    auto serializer(armnnSerializer::ISerializer::Create());
    // Serialize the Network
    serializer->Serialize(network);
    if (dataCachingActive)
    {
        std::stringstream stream;
        auto serialized = serializer->SaveSerializedToStream(stream);
        if (serialized)
        {
            std::string const serializedString{stream.str()};
            std::copy(serializedString.begin(),
                      serializedString.end(),
                      std::back_inserter(dataCacheData));
        }
    }

    if (bSerializeToFile)
    {
        // Set the name of the output .armnn file.
        fs::path dumpPath = dumpDir;
        std::string timestamp = GetFileTimestamp();
        fs::path tempFilePath = dumpPath / (timestamp + "_network.armnn");
        fileName = tempFilePath.string();

        // Save serialized network to a file
        std::ofstream serializedFile(fileName, std::ios::out | std::ios::binary);
        auto serialized = serializer->SaveSerializedToStream(serializedFile);
        if (!serialized)
        {
            VLOG(DRIVER) << "An error occurred when serializing to file %s" << fileName.c_str();
        }
    }
    return fileName;
}

bool IsDynamicTensor(const armnn::TensorInfo& tensorInfo)
{
    if (tensorInfo.GetShape().GetDimensionality() == armnn::Dimensionality::NotSpecified)
    {
        return true;
    }
    // Account for the usage of the TensorShape empty constructor
    if (tensorInfo.GetNumDimensions() == 0)
    {
        return true;
    }
    return !tensorInfo.GetShape().AreAllDimensionsSpecified();
}

bool AreDynamicTensorsSupported() //TODO
{
    return true;
}

bool isQuantizedOperand(const OperandType& operandType)
{
    if (operandType == OperandType::TENSOR_QUANT8_ASYMM ||
        operandType == OperandType::TENSOR_QUANT8_SYMM_PER_CHANNEL ||
        operandType == OperandType::TENSOR_QUANT8_SYMM ||
        operandType == OperandType::TENSOR_QUANT16_SYMM ||
        operandType == OperandType::TENSOR_QUANT8_ASYMM_SIGNED)
    {
        return true;
    }
    else
    {
        return false;
    }
}

std::string GetModelSummary(const Model& model)
{
    std::stringstream result;

    result << model.main.inputIndexes.size() << " input(s), "
           << model.main.operations.size() << " operation(s), "
           << model.main.outputIndexes.size() << " output(s), "
           << model.main.operands.size() << " operand(s) "
           << std::endl;

    result << "Inputs: ";
    for (uint32_t i = 0; i < model.main.inputIndexes.size(); i++)
    {
        result << GetOperandSummary(model.main.operands[model.main.inputIndexes[i]]) << ", ";
    }
    result << std::endl;

    result << "Operations: ";
    for (uint32_t i = 0; i < model.main.operations.size(); i++)
    {
        result << model.main.operations[i].type << ", ";
    }
    result << std::endl;

    result << "Outputs: ";
    for (uint32_t i = 0; i < model.main.outputIndexes.size(); i++)
    {
        result << GetOperandSummary(model.main.operands[model.main.outputIndexes[i]]) << ", ";
    }
    result << std::endl;

    return result.str();
}

std::string GetFileTimestamp()
{
    // used to get a timestamp to name diagnostic files (the ArmNN serialized graph
    // and getSupportedOperations.txt files)
    timespec ts;
    int iRet = clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    std::stringstream ss;
    if (iRet == 0)
    {
        ss << std::to_string(ts.tv_sec) << "_" << std::to_string(ts.tv_nsec);
    }
    else
    {
        VLOG(DRIVER) << "clock_gettime failed with errno " <<
            std::to_string(errno).c_str() << " : " <<
            std::strerror(errno);
    }
    return ss.str();
}

void RenameExportedFiles(const std::string& existingSerializedFileName,
                         const std::string& existingDotFileName,
                         const std::string& dumpDir,
                         const armnn::NetworkId networkId)
{
    if (dumpDir.empty())
    {
        return;
    }
    RenameFile(existingSerializedFileName, std::string("_network.armnn"), dumpDir, networkId);
    RenameFile(existingDotFileName, std::string("_networkgraph.dot"), dumpDir, networkId);
}

void RenameFile(const std::string& existingName,
                const std::string& extension,
                const std::string& dumpDir,
                const armnn::NetworkId networkId)
{
    if (existingName.empty() || dumpDir.empty())
    {
        return;
    }

    fs::path dumpPath = dumpDir;
    const fs::path newFileName = dumpPath / (std::to_string(networkId) + extension);
    int iRet = rename(existingName.c_str(), newFileName.c_str());
    if (iRet != 0)
    {
        std::stringstream ss;
        ss << "rename of [" << existingName << "] to [" << newFileName << "] failed with errno "
           << std::to_string(errno) << " : " << std::strerror(errno);
        VLOG(DRIVER) << ss.str().c_str();
    }
}

void CommitPools(std::vector<::android::nn::RunTimePoolInfo>& memPools)
{
    // Commit output buffers.
    // Note that we update *all* pools, even if they aren't actually used as outputs -
    // this is simpler and is what the CpuExecutor does.
    for (auto& pool : memPools)
    {
        // Type android::nn::RunTimePoolInfo has changed between Android P & Q and Android R, where
        // update() has been removed and flush() added.
        pool.flush();
    }
}
} // namespace armnn_driver
