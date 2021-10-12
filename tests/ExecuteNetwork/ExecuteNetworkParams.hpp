//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/BackendId.hpp>
#include <armnn/Tensor.hpp>

#if defined(ARMNN_TFLITE_DELEGATE)
#include <DelegateOptions.hpp>
#endif

/// Holds all parameters necessary to execute a network
/// Check ExecuteNetworkProgramOptions.cpp for a description of each parameter
struct ExecuteNetworkParams
{
    using TensorShapePtr = std::unique_ptr<armnn::TensorShape>;

    enum class TfLiteExecutor
    {
        ArmNNTfLiteParser,
        ArmNNTfLiteDelegate,
        TfliteInterpreter
    };

    std::string                   m_CachedNetworkFilePath;
    std::vector<armnn::BackendId> m_ComputeDevices;
    bool                          m_Concurrent;
    bool                          m_DequantizeOutput;
    std::string                   m_DynamicBackendsPath;
    bool                          m_EnableBf16TurboMode;
    bool                          m_EnableFastMath = false;
    bool                          m_EnableFp16TurboMode;
    bool                          m_EnableLayerDetails = false;
    bool                          m_EnableProfiling;
    bool                          m_GenerateTensorData;
    bool                          m_InferOutputShape = false;
    bool                          m_EnableDelegate = false;
    std::vector<std::string>      m_InputNames;
    std::vector<std::string>      m_InputTensorDataFilePaths;
    std::vector<TensorShapePtr>   m_InputTensorShapes;
    std::vector<std::string>      m_InputTypes;
    bool                          m_IsModelBinary;
    size_t                        m_Iterations;
    std::string                   m_ModelFormat;
    std::string                   m_ModelPath;
    unsigned int                  m_NumberOfThreads;
    bool                          m_OutputDetailsToStdOut;
    bool                          m_OutputDetailsOnlyToStdOut;
    std::vector<std::string>      m_OutputNames;
    std::vector<std::string>      m_OutputTensorFiles;
    std::vector<std::string>      m_OutputTypes;
    bool                          m_ParseUnsupported = false;
    bool                          m_PrintIntermediate;
    bool                          m_DontPrintOutputs;
    bool                          m_QuantizeInput;
    bool                          m_SaveCachedNetwork;
    size_t                        m_SimultaneousIterations;
    size_t                        m_SubgraphId;
    double                        m_ThresholdTime;
    int                           m_TuningLevel;
    std::string                   m_TuningPath;
    std::string                   m_MLGOTuningFilePath;
    TfLiteExecutor                m_TfLiteExecutor;
    size_t                        m_ThreadPoolSize;

    // Ensures that the parameters for ExecuteNetwork fit together
    void ValidateParams();

#if defined(ARMNN_TFLITE_DELEGATE)
    /// A utility method that populates a DelegateOptions object from this ExecuteNetworkParams.
    armnnDelegate::DelegateOptions ToDelegateOptions() const;
#endif

};
