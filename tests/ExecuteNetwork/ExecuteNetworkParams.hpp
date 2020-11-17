//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/BackendId.hpp>
#include <armnn/Tensor.hpp>

/// Holds all parameters necessary to execute a network
/// Check ExecuteNetworkProgramOptions.cpp for a description of each parameter
struct ExecuteNetworkParams
{
    using TensorShapePtr = std::unique_ptr<armnn::TensorShape>;

    std::vector<armnn::BackendId> m_ComputeDevices;
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
    std::vector<std::string>      m_OutputNames;
    std::vector<std::string>      m_OutputTensorFiles;
    std::vector<std::string>      m_OutputTypes;
    bool                          m_ParseUnsupported = false;
    bool                          m_PrintIntermediate;
    bool                          m_QuantizeInput;
    size_t                        m_SubgraphId;
    double                        m_ThresholdTime;
    int                           m_TuningLevel;
    std::string                   m_TuningPath;

    // Ensures that the parameters for ExecuteNetwork fit together
    void ValidateParams();
};
