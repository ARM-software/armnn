//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include "TestBlocks.hpp"
#include "TestStrategy.hpp"

#include <IMemoryOptimizerStrategy.hpp>
#include <MemoryOptimizerStrategyLibrary.hpp>
#include <strategies/StrategyValidator.hpp>


#include <cxxopts.hpp>

#include <iostream>
#include <algorithm>
#include <iomanip>

std::vector<TestBlock> testBlocks
{
    {"fsrcnn", fsrcnn},
    {"inceptionv4", inceptionv4},
    {"deeplabv3", deeplabv3},
    {"deepspeechv1", deepspeechv1},
    {"mobilebert", mobilebert},
    {"ssd_mobilenetv2", ssd_mobilenetv2},
    {"resnetv2", resnetv2},
    {"yolov3",yolov3}
};

void PrintModels()
{
    std::cout << "Available models:\n";
    for (const auto& model : testBlocks)
    {
        std::cout << model.m_Name << "\n";
    }
    std::cout << "\n";
}

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

void RunBenchmark(armnn::IMemoryOptimizerStrategy* strategy, std::vector<TestBlock>* models)
{
    using Clock = std::chrono::high_resolution_clock;
    float avgEfficiency = 0;
    std::chrono::duration<double, std::milli> avgDuration{};
    std::cout << "\nMemory Strategy: "  << strategy->GetName()<< "\n";
    std::cout << "===============================================\n";
    for (auto& model : *models)
    {
        auto now = Clock::now();
        const std::vector<armnn::MemBin> result = strategy->Optimize(model.m_Blocks);
        auto duration = std::chrono::duration<double, std::milli>(Clock::now() - now);

        avgDuration += duration;
        size_t memoryUsage = 0;
        for (auto bin : result)
        {
            memoryUsage += bin.m_MemSize;
        }
        size_t minSize = GetMinPossibleMemorySize(model.m_Blocks);

        float efficiency = static_cast<float>(minSize) / static_cast<float>(memoryUsage);
        efficiency*=100;
        avgEfficiency += efficiency;
        std::cout << "\nModel: " << model.m_Name << "\n";

        std::cout << "Strategy execution time: " << std::setprecision(4) << duration.count() << " milliseconds\n";

        std::cout << "Memory usage: " << memoryUsage/1024 << " kb\n";

        std::cout << "Minimum possible usage: " << minSize/1024 << " kb\n";

        std::cout << "Memory efficiency: " << std::setprecision(3) << efficiency << "%\n";
    }

    avgDuration/= static_cast<double>(models->size());
    avgEfficiency/= static_cast<float>(models->size());

    std::cout << "\n===============================================\n";
    std::cout << "Average memory duration: " << std::setprecision(4) << avgDuration.count() << " milliseconds\n";
    std::cout << "Average memory efficiency: " << std::setprecision(3) << avgEfficiency << "%\n";
}

struct BenchmarkOptions
{
    std::string m_StrategyName;
    std::string m_ModelName;
    bool m_UseDefaultStrategy = false;
    bool m_Validate = false;
};

BenchmarkOptions ParseOptions(int argc, char* argv[])
{
    cxxopts::Options options("Memory Benchmark", "Tests memory optimization strategies on different models");

    options.add_options()
        ("s, strategy", "Strategy name, do not specify to use default strategy", cxxopts::value<std::string>())
        ("m, model", "Model name", cxxopts::value<std::string>())
        ("v, validate", "Validate strategy", cxxopts::value<bool>()->default_value("false")->implicit_value("true"))
        ("h,help", "Display usage information");

    auto result = options.parse(argc, argv);
    if (result.count("help"))
    {
        std::cout << options.help() << std::endl;
        PrintModels();

        std::cout << "\nAvailable strategies:\n";

        for (const auto& s :armnn::GetMemoryOptimizerStrategyNames())
        {
            std::cout << s << "\n";
        }
        exit(EXIT_SUCCESS);
    }

    BenchmarkOptions benchmarkOptions;

    if(result.count("strategy"))
    {
        benchmarkOptions.m_StrategyName = result["strategy"].as<std::string>();
    }
    else
    {
        std::cout << "No Strategy given, using default strategy";

        benchmarkOptions.m_UseDefaultStrategy = true;
    }

    if(result.count("model"))
    {
        benchmarkOptions.m_ModelName = result["model"].as<std::string>();
    }

    benchmarkOptions.m_Validate = result["validate"].as<bool>();

    return benchmarkOptions;
}

int main(int argc, char* argv[])
{
    BenchmarkOptions benchmarkOptions = ParseOptions(argc, argv);

    std::shared_ptr<armnn::IMemoryOptimizerStrategy> strategy;

    if (benchmarkOptions.m_UseDefaultStrategy)
    {
        strategy = std::make_shared<armnn::TestStrategy>();
    }
    else
    {
        strategy = armnn::GetMemoryOptimizerStrategy(benchmarkOptions.m_StrategyName);

        if (!strategy)
        {
            std::cout << "Strategy name not found\n";
            return 0;
        }
    }

    std::vector<TestBlock> model;
    std::vector<TestBlock>* modelsToTest = &testBlocks;
    if (benchmarkOptions.m_ModelName.size() != 0)
    {
        auto it = std::find_if(testBlocks.cbegin(), testBlocks.cend(), [&](const TestBlock testBlock)
        {
            return testBlock.m_Name == benchmarkOptions.m_ModelName;
        });

        if (it == testBlocks.end())
        {
            std::cout << "Model name not found\n";
            return 0;
        }
        else
        {
            model.push_back(*it);
            modelsToTest = &model;
        }
    }

    if (benchmarkOptions.m_Validate)
    {
        armnn::StrategyValidator strategyValidator;

        strategyValidator.SetStrategy(strategy);

        RunBenchmark(&strategyValidator, modelsToTest);
    }
    else
    {
        RunBenchmark(strategy.get(), modelsToTest);
    }

}