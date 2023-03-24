//
// Copyright © 2017, 2023 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ClWorkloadFactoryHelper.hpp"

#include <Network.hpp>

#include <GraphUtils.hpp>

#include <cl/ClWorkloadFactory.hpp>
#include <cl/ClBackendContext.hpp>

#include <armnnUtils/Filesystem.hpp>

#include <doctest/doctest.h>

TEST_SUITE("ClOptimizedNetwork")
{
TEST_CASE("OptimizeValidateGpuDeviceSupportLayerNoFallback")
{
    // build up the structure of the network
    armnn::INetworkPtr net(armnn::INetwork::Create());

    armnn::IConnectableLayer* input  = net->AddInputLayer(0);
    armnn::IConnectableLayer* output = net->AddOutputLayer(0);

    input->GetOutputSlot(0).Connect(output->GetInputSlot(0));
    input->GetOutputSlot(0).SetTensorInfo(armnn::TensorInfo({ 1, 1, 4, 4 }, armnn::DataType::Float32));

    armnn::IRuntime::CreationOptions options;
    armnn::IRuntimePtr runtime(armnn::IRuntime::Create(options));

    std::vector<armnn::BackendId> backends = { armnn::Compute::GpuAcc };
    armnn::IOptimizedNetworkPtr optNet = armnn::Optimize(*net, backends, runtime->GetDeviceSpec());
    CHECK(optNet);
    // validate workloads
    armnn::ClWorkloadFactory fact =
        ClWorkloadFactoryHelper::GetFactory(ClWorkloadFactoryHelper::GetMemoryManager());

    const armnn::Graph& theGraph = GetGraphForTesting(optNet.get());
    for (auto&& layer : theGraph)
    {
        CHECK(layer->GetBackendId() == armnn::Compute::GpuAcc);
        CHECK_NOTHROW(
            layer->CreateWorkload(fact));
    }
}

TEST_CASE("FP16TurboModeTestOnGpuAcc")
{
    // Test to check when Fp16 Turbo mode set
    // it converts the Fp32 network to Fp16 Network
    // add Fp32ToFp16 conversion layer after the InputLayer
    // add Fp16ToFp32 conversion layer after the OutputLayer
    // checks the other layers if they are supported in Fp16
    // if they are not put the conversion layers before and after
    // if they are not supported in Fp16 use Fp32 instead
    // if there are inverse conversion layers remove them with optimization
    // at the moment FloorLayer is not supported in Fp16 so it rolls back to Fp32
    // and inverse conversion layers are removed by the optimizer
    armnn::INetworkPtr net(armnn::INetwork::Create());

    // Defines layers.
    auto input = net->AddInputLayer(0, "input layer");
    // ReLu1
    armnn::ActivationDescriptor activation1Descriptor;
    activation1Descriptor.m_Function = armnn::ActivationFunction::BoundedReLu;
    activation1Descriptor.m_A = 1.f;
    activation1Descriptor.m_B = -1.f;
    auto activation = net->AddActivationLayer(activation1Descriptor, "activation layer");
    auto output = net->AddOutputLayer(0, "output layer");

    // Connects layers.
    input->GetOutputSlot(0).Connect(activation->GetInputSlot(0));
    activation->GetOutputSlot(0).Connect(output->GetInputSlot(0));

    armnn::TensorShape shape({4});
    armnn::TensorInfo info(shape, armnn::DataType::Float32);
    input->GetOutputSlot(0).SetTensorInfo(info);
    activation->GetOutputSlot(0).SetTensorInfo(info);

    armnn::IRuntime::CreationOptions options;
    armnn::IRuntimePtr runtime(armnn::IRuntime::Create(options));

    std::vector<armnn::BackendId> backends = {armnn::Compute::GpuAcc};

    armnn::OptimizerOptionsOpaque optimizerOptions;
    optimizerOptions.SetReduceFp32ToFp16(true);

    armnn::IOptimizedNetworkPtr optimizedNet = armnn::Optimize(
            *net, backends, runtime->GetDeviceSpec(), optimizerOptions);

    const armnn::Graph& graph = GetGraphForTesting(optimizedNet.get());

    // Tests that all layers are present in the graph.
    CHECK(graph.GetNumLayers() == 5);

    // Tests that the vertices exist and have correct names.
    CHECK(GraphHasNamedLayer(graph, "input layer"));
    CHECK(GraphHasNamedLayer(graph, "convert_fp32_to_fp16-0-input layer"));
    CHECK(GraphHasNamedLayer(graph, "activation layer"));
    CHECK(GraphHasNamedLayer(graph, "convert_fp16_to_fp32-0-output layer"));
    CHECK(GraphHasNamedLayer(graph, "output layer"));
}

TEST_CASE("FastMathEnabledTestOnGpuAcc")
{
    armnn::INetworkPtr net(armnn::INetwork::Create());

    armnn::IConnectableLayer* input  = net->AddInputLayer(0);
    armnn::IConnectableLayer* output = net->AddOutputLayer(0);

    input->GetOutputSlot(0).Connect(output->GetInputSlot(0));
    input->GetOutputSlot(0).SetTensorInfo(armnn::TensorInfo({ 1, 1, 4, 4 }, armnn::DataType::Float32));

    armnn::IRuntime::CreationOptions options;
    armnn::IRuntimePtr runtime(armnn::IRuntime::Create(options));

    std::vector<armnn::BackendId> backends = {armnn::Compute::GpuAcc};
    armnn::OptimizerOptionsOpaque optimizerOptions;
    armnn::BackendOptions modelOptions("GpuAcc", {{"FastMathEnabled", true}});
    optimizerOptions.AddModelOption(modelOptions);

    armnn::IOptimizedNetworkPtr optimizedNet = armnn::Optimize(
    *net, backends, runtime->GetDeviceSpec(), optimizerOptions);

    CHECK(optimizedNet);

    auto modelOptionsOut = GetModelOptionsForTesting(optimizedNet.get());

    CHECK(modelOptionsOut.size() == 2); // FastMathEnabled and the Global to hold the import export values.
    CHECK(modelOptionsOut[0].GetOption(0).GetName() == "FastMathEnabled");
    CHECK(modelOptionsOut[0].GetOption(0).GetValue().AsBool() == true);
}

TEST_CASE("CheckMLGOTuningFile")
{
    class ClBackendContextTestClass : public armnn::ClBackendContext
    {
    public:
        ClBackendContextTestClass(const armnn::IRuntime::CreationOptions &options) : ClBackendContext(options)
        {}

        bool call_reload_from_file()
        {
            return m_MLGOTuner.reload_from_file(m_MLGOTuningFile);
        }
    };

    const std::string validText{
            "<header>\n"
            "gemm-version, [1,2,1]\n"
            "ip-type,gpu\n"
            "</header>\n"
            "<heuristics-table>\n"
            "0, g71 , 8, f32, best-performance, static, gemm-type, [m,n,k,n]\n"
            "1, g71 , 8, f32, best-performance, static, gemm-config-reshaped-only-rhs, [m,n,k,n]\n"
            "2, g71 , 8, f32, best-performance, static, gemm-config-reshaped, [m,n,k,n]\n"
            "3, g71 , 8, qasymm8, best-performance, static, gemm-type, [m,n,k,n]\n"
            "4, g71 , 8, qasymm8, best-performance, static, gemm-config-reshaped-only-rhs, [m,n,k,n]\n"
            "5, g71 , 8, qasymm8, best-performance, static, gemm-config-native, [m,n,k,n]\n"
            "</heuristics-table>\n"
            "<heuristic, 0>\n"
            "b , 0, var, r_mn, >=, num, 2., 1, 2\n"
            "l , 1, gemm-type, reshaped\n"
            "l , 2, gemm-type, reshaped-only-rhs\n"
            "</heuristic>\n"
            "<heuristic, 1>\n"
            "l ,0,gemm-config-reshaped-only-rhs, [2, 4,4,4,1,1,0]\n"
            "</heuristic>\n"
            "<heuristic, 2>\n"
            "l ,0,gemm-config-reshaped,[4,2,8,16,16,1,0,1,0]\n"
            "</heuristic>\n"
            "<heuristic, 3>\n"
            "l , 0, gemm-type, native\n"
            "</heuristic>\n"
            "<heuristic, 4>\n"
            "l ,0,gemm-config-reshaped-only-rhs, [2, 4,4,4,1,1,0]\n"
            "</heuristic>\n"
            "<heuristic, 5>\n"
            "l ,0,gemm-config-native,[4,2,8]\n"
            "</heuristic>\n"};

    const std::string invalidText{"ʕノ•ᴥ•ʔノ ︵ ┻━┻"};

    fs::path validFile = armnnUtils::Filesystem::NamedTempFile("validFile.mlgo");
    fs::path invalidFile = armnnUtils::Filesystem::NamedTempFile("invalidFile.mlgo");

    try
    {
        std::ofstream ofs1{validFile};
        ofs1 << validText << std::endl;
        ofs1.close();

        std::ofstream ofs2{invalidFile};
        ofs2 << invalidText << std::endl;
        ofs2.close();
    }
    catch (std::exception &e)
    {
        std::cerr << "Unable to write to file at location [" << validFile.c_str() << "] : " << e.what() << std::endl;
        CHECK(false);
    }

    armnn::IRuntime::CreationOptions creationOptions1;
    armnn::BackendOptions validOptions
            {
                    "GpuAcc",
                    {
                            {"MLGOTuningFilePath", validFile.c_str()}
                    }
            };

    creationOptions1.m_BackendOptions.emplace_back(validOptions);
    ClBackendContextTestClass clBackendContext1(creationOptions1);
    CHECK(clBackendContext1.call_reload_from_file());

    armnn::BackendOptions invalidOptions
            {
                    "GpuAcc",
                    {
                            {"MLGOTuningFilePath", invalidFile.c_str()}
                    }
            };

    armnn::IRuntime::CreationOptions creationOptions2;
    creationOptions2.m_BackendOptions.emplace_back(invalidOptions);
    ClBackendContextTestClass clBackendContext2(creationOptions2);
    CHECK(clBackendContext2.call_reload_from_file() == false);

    armnn::BackendOptions invalidPathOptions
            {
                    "GpuAcc",
                    {
                            {"MLGOTuningFilePath", "not_a_real_file_path"}
                    }
            };

    armnn::IRuntime::CreationOptions creationOptions3;
    creationOptions3.m_BackendOptions.emplace_back(invalidPathOptions);
    ClBackendContextTestClass clBackendContext3(creationOptions3);
    CHECK(clBackendContext3.call_reload_from_file() == false);
}

}
