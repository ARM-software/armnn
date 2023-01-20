//
// Copyright © 2022-2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "IExecutor.hpp"
#include "NetworkExecutionUtils/NetworkExecutionUtils.hpp"
#include "ExecuteNetworkProgramOptions.hpp"
#include "armnn/utility/NumericCast.hpp"
#include "armnn/utility/Timer.hpp"

#include <armnn/ArmNN.hpp>
#include <armnn/Threadpool.hpp>
#include <armnn/Logging.hpp>
#include <armnn/utility/Timer.hpp>
#include <armnn/BackendRegistry.hpp>
#include <armnn/utility/Assert.hpp>
#include <armnn/utility/NumericCast.hpp>

#include <armnnUtils/Filesystem.hpp>
#include <HeapProfiling.hpp>

#include <fmt/format.h>

#if defined(ARMNN_SERIALIZER)
#include "armnnDeserializer/IDeserializer.hpp"
#endif
#if defined(ARMNN_TF_LITE_PARSER)
#include <armnnTfLiteParser/ITfLiteParser.hpp>
#endif
#if defined(ARMNN_ONNX_PARSER)
#include <armnnOnnxParser/IOnnxParser.hpp>
#endif

class ArmNNExecutor : public IExecutor
{
public:
    ArmNNExecutor(const ExecuteNetworkParams& params, armnn::IRuntime::CreationOptions runtimeOptions);

    std::vector<const void* > Execute() override;
    void PrintNetworkInfo() override;
    void CompareAndPrintResult(std::vector<const void*> otherOutput) override;

private:

    /**
     * Returns a pointer to the armnn::IRuntime* this will be shared by all ArmNNExecutors.
     */
    armnn::IRuntime* GetRuntime(const armnn::IRuntime::CreationOptions& options)
    {
        static armnn::IRuntimePtr instance = armnn::IRuntime::Create(options);
        // Instantiated on first use.
        return instance.get();
    }

    struct IParser;
    struct IOInfo;
    struct IOStorage;

    using BindingPointInfo = armnn::BindingPointInfo;

    std::unique_ptr<IParser> CreateParser();

    void ExecuteAsync();
    void ExecuteSync();
    void SetupInputsAndOutputs();

    IOInfo GetIOInfo(armnn::IOptimizedNetwork* optNet);

    void PrintOutputTensors(const armnn::OutputTensors* outputTensors, unsigned int iteration);

    armnn::IOptimizedNetworkPtr OptimizeNetwork(armnn::INetwork* network);

    struct IOStorage
    {
        IOStorage(size_t size)
        {
            m_Mem = operator new(size);
        }
        ~IOStorage()
        {
            operator delete(m_Mem);
        }
        IOStorage(IOStorage&& rhs)
        {
            this->m_Mem = rhs.m_Mem;
            rhs.m_Mem = nullptr;
        }

        IOStorage(const IOStorage& rhs) = delete;
        IOStorage& operator=(IOStorage& rhs) = delete;
        IOStorage& operator=(IOStorage&& rhs) = delete;

        void* m_Mem;
    };

    struct IOInfo
    {
        std::vector<std::string> m_InputNames;
        std::vector<std::string> m_OutputNames;
        std::map<std::string, armnn::BindingPointInfo> m_InputInfoMap;
        std::map<std::string, armnn::BindingPointInfo> m_OutputInfoMap;
    };

    IOInfo m_IOInfo;
    std::vector<IOStorage> m_InputStorage;
    std::vector<IOStorage> m_OutputStorage;
    std::vector<armnn::InputTensors> m_InputTensorsVec;
    std::vector<armnn::OutputTensors> m_OutputTensorsVec;
    std::vector<std::vector<unsigned int>> m_ImportedInputIds;
    std::vector<std::vector<unsigned int>> m_ImportedOutputIds;
    armnn::IRuntime* m_Runtime;
    armnn::NetworkId m_NetworkId;
    ExecuteNetworkParams m_Params;

    struct IParser
    {
        virtual armnn::INetworkPtr CreateNetwork(const ExecuteNetworkParams& params) = 0;
        virtual armnn::BindingPointInfo GetInputBindingPointInfo(size_t id, const std::string& inputName) = 0;
        virtual armnn::BindingPointInfo GetOutputBindingPointInfo(size_t id, const std::string& outputName) = 0;

        virtual ~IParser(){};
    };

#if defined(ARMNN_SERIALIZER)
    class ArmNNDeserializer : public IParser
    {
        public:
        ArmNNDeserializer();

        armnn::INetworkPtr CreateNetwork(const ExecuteNetworkParams& params) override;
        armnn::BindingPointInfo GetInputBindingPointInfo(size_t, const std::string& inputName) override;
        armnn::BindingPointInfo GetOutputBindingPointInfo(size_t, const std::string& outputName) override;

        private:
        armnnDeserializer::IDeserializerPtr m_Parser;
    };
#endif

#if defined(ARMNN_TF_LITE_PARSER)
    class TfliteParser : public IParser
    {
    public:
        TfliteParser(const ExecuteNetworkParams& params);

        armnn::INetworkPtr CreateNetwork(const ExecuteNetworkParams& params) override;
        armnn::BindingPointInfo GetInputBindingPointInfo(size_t subgraphId, const std::string& inputName) override;
        armnn::BindingPointInfo GetOutputBindingPointInfo(size_t subgraphId, const std::string& outputName) override;

    private:
        armnnTfLiteParser::ITfLiteParserPtr m_Parser{nullptr, [](armnnTfLiteParser::ITfLiteParser*){}};
    };
#endif

#if defined(ARMNN_ONNX_PARSER)
    class OnnxParser : public IParser
    {
        public:
        OnnxParser();

        armnn::INetworkPtr CreateNetwork(const ExecuteNetworkParams& params) override;
        armnn::BindingPointInfo GetInputBindingPointInfo(size_t subgraphId, const std::string& inputName) override;
        armnn::BindingPointInfo GetOutputBindingPointInfo(size_t subgraphId, const std::string& outputName) override;

        private:
        armnnOnnxParser::IOnnxParserPtr m_Parser;
    };
#endif
};