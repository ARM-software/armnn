//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once
#include <armnn/ArmNN.hpp>

#include <CpuExecutor.h>
#include <nnapi/OperandTypes.h>
#include <nnapi/Result.h>
#include <nnapi/Types.h>

#include <vector>
#include <string>
#include <fstream>
#include <iomanip>

namespace armnn_driver
{

using namespace android::nn;

extern const armnn::PermutationVector g_DontPermute;

template <typename OperandType>
class UnsupportedOperand: public std::runtime_error
{
public:
    UnsupportedOperand(const OperandType type)
        : std::runtime_error("Operand type is unsupported")
        , m_type(type)
    {}

    OperandType m_type;
};

/// Swizzles tensor data in @a input according to the dimension mappings.
void SwizzleAndroidNn4dTensorToArmNn(armnn::TensorInfo& tensor,
                                     const void* input,
                                     void* output,
                                     const armnn::PermutationVector& mappings);

/// Returns a pointer to a specific location in a pool`
void* GetMemoryFromPool(DataLocation location,
                        const std::vector<android::nn::RunTimePoolInfo>& memPools);

void* GetMemoryFromPointer(const Request::Argument& requestArg);

armnn::TensorInfo GetTensorInfoForOperand(const Operand& operand);

std::string GetOperandSummary(const Operand& operand);

bool isQuantizedOperand(const OperandType& operandType);

std::string GetModelSummary(const Model& model);

template <typename TensorType>
void DumpTensor(const std::string& dumpDir,
                const std::string& requestName,
                const std::string& tensorName,
                const TensorType& tensor);

void DumpJsonProfilingIfRequired(bool gpuProfilingEnabled,
                                 const std::string& dumpDir,
                                 armnn::NetworkId networkId,
                                 const armnn::IProfiler* profiler);

std::string ExportNetworkGraphToDotFile(const armnn::IOptimizedNetwork& optimizedNetwork,
                                        const std::string& dumpDir);

std::string SerializeNetwork(const armnn::INetwork& network,
                             const std::string& dumpDir,
                             std::vector<uint8_t>& dataCacheData,
                             bool dataCachingActive = true);

void RenameExportedFiles(const std::string& existingSerializedFileName,
                         const std::string& existingDotFileName,
                         const std::string& dumpDir,
                         const armnn::NetworkId networkId);

void RenameFile(const std::string& existingName,
                const std::string& extension,
                const std::string& dumpDir,
                const armnn::NetworkId networkId);

/// Checks if a tensor info represents a dynamic tensor
bool IsDynamicTensor(const armnn::TensorInfo& outputInfo);

/// Checks for ArmNN support of dynamic tensors.
bool AreDynamicTensorsSupported(void);

std::string GetFileTimestamp();

inline OutputShape ComputeShape(const armnn::TensorInfo& info)
{
    OutputShape shape;

    armnn::TensorShape tensorShape = info.GetShape();
    // Android will expect scalars as a zero dimensional tensor
    if(tensorShape.GetDimensionality() == armnn::Dimensionality::Scalar)
    {
         shape.dimensions = std::vector<uint32_t>{};
    }
    else
    {
        std::vector<uint32_t> dimensions;
        const unsigned int numDims = tensorShape.GetNumDimensions();
        dimensions.resize(numDims);
        for (unsigned int outputIdx = 0u; outputIdx < numDims; ++outputIdx)
        {
            dimensions[outputIdx] = tensorShape[outputIdx];
        }
        shape.dimensions = dimensions;
    }

    shape.isSufficient = true;

    return shape;
}

void CommitPools(std::vector<::android::nn::RunTimePoolInfo>& memPools);

} // namespace armnn_driver
