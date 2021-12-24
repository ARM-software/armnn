//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "MockBackendId.hpp"
#include "armnn/backends/profiling/IBackendProfiling.hpp"
#include "armnn/backends/profiling/IBackendProfilingContext.hpp"

#include <LayerSupportCommon.hpp>
#include <armnn/backends/IBackendInternal.hpp>
#include <armnn/backends/OptimizationViews.hpp>
#include <armnn/backends/profiling/IBackendProfiling.hpp>
#include <backends/BackendProfiling.hpp>
#include <backendsCommon/LayerSupportBase.hpp>

namespace armnn
{

class MockBackendInitialiser
{
public:
    MockBackendInitialiser();
    ~MockBackendInitialiser();
};

class MockBackendProfilingContext : public profiling::IBackendProfilingContext
{
public:
    MockBackendProfilingContext(IBackendInternal::IBackendProfilingPtr& backendProfiling)
        : m_BackendProfiling(std::move(backendProfiling))
        , m_CapturePeriod(0)
        , m_IsTimelineEnabled(true)
    {}

    ~MockBackendProfilingContext() = default;

    IBackendInternal::IBackendProfilingPtr& GetBackendProfiling()
    {
        return m_BackendProfiling;
    }

    uint16_t RegisterCounters(uint16_t currentMaxGlobalCounterId)
    {
        std::unique_ptr<profiling::IRegisterBackendCounters> counterRegistrar =
            m_BackendProfiling->GetCounterRegistrationInterface(static_cast<uint16_t>(currentMaxGlobalCounterId));

        std::string categoryName("MockCounters");
        counterRegistrar->RegisterCategory(categoryName);

        counterRegistrar->RegisterCounter(0, categoryName, 0, 0, 1.f, "Mock Counter One", "Some notional counter");

        counterRegistrar->RegisterCounter(1, categoryName, 0, 0, 1.f, "Mock Counter Two",
                                                                   "Another notional counter");

        std::string units("microseconds");
        uint16_t nextMaxGlobalCounterId =
                counterRegistrar->RegisterCounter(2, categoryName, 0, 0, 1.f, "Mock MultiCore Counter",
                                                                   "A dummy four core counter", units, 4);
        return nextMaxGlobalCounterId;
    }

    Optional<std::string> ActivateCounters(uint32_t capturePeriod, const std::vector<uint16_t>& counterIds)
    {
        if (capturePeriod == 0 || counterIds.size() == 0)
        {
            m_ActiveCounters.clear();
        }
        else if (capturePeriod == 15939u)
        {
            return armnn::Optional<std::string>("ActivateCounters example test error");
        }
        m_CapturePeriod  = capturePeriod;
        m_ActiveCounters = counterIds;
        return armnn::Optional<std::string>();
    }

    std::vector<profiling::Timestamp> ReportCounterValues()
    {
        std::vector<profiling::CounterValue> counterValues;

        for (auto counterId : m_ActiveCounters)
        {
            counterValues.emplace_back(profiling::CounterValue{ counterId, counterId + 1u });
        }

        uint64_t timestamp = m_CapturePeriod;
        return { profiling::Timestamp{ timestamp, counterValues } };
    }

    bool EnableProfiling(bool)
    {
        auto sendTimelinePacket = m_BackendProfiling->GetSendTimelinePacket();
        sendTimelinePacket->SendTimelineEntityBinaryPacket(4256);
        sendTimelinePacket->Commit();
        return true;
    }

    bool EnableTimelineReporting(bool isEnabled)
    {
        m_IsTimelineEnabled = isEnabled;
        return isEnabled;
    }

    bool TimelineReportingEnabled()
    {
        return m_IsTimelineEnabled;
    }

private:
    IBackendInternal::IBackendProfilingPtr m_BackendProfiling;
    uint32_t m_CapturePeriod;
    std::vector<uint16_t> m_ActiveCounters;
    std::atomic<bool> m_IsTimelineEnabled;
};

class MockBackendProfilingService
{
public:
    // Getter for the singleton instance
    static MockBackendProfilingService& Instance()
    {
        static MockBackendProfilingService instance;
        return instance;
    }

    MockBackendProfilingContext* GetContext()
    {
        return m_sharedContext.get();
    }

    void SetProfilingContextPtr(std::shared_ptr<MockBackendProfilingContext> shared)
    {
        m_sharedContext = shared;
    }

private:
    std::shared_ptr<MockBackendProfilingContext> m_sharedContext;
};

class MockBackend : public IBackendInternal
{
public:
    MockBackend()  = default;
    ~MockBackend() = default;

    static const BackendId& GetIdStatic();
    const BackendId& GetId() const override
    {
        return GetIdStatic();
    }

    IBackendInternal::IMemoryManagerUniquePtr CreateMemoryManager() const override;

    IBackendInternal::IWorkloadFactoryPtr
        CreateWorkloadFactory(const IBackendInternal::IMemoryManagerSharedPtr& memoryManager = nullptr) const override;

    IBackendInternal::IBackendContextPtr CreateBackendContext(const IRuntime::CreationOptions&) const override;
    IBackendInternal::IBackendProfilingContextPtr
        CreateBackendProfilingContext(const IRuntime::CreationOptions& creationOptions,
                                      IBackendProfilingPtr& backendProfiling) override;

    IBackendInternal::ILayerSupportSharedPtr GetLayerSupport() const override;

    OptimizationViews OptimizeSubgraphView(const SubgraphView& subgraph) const override;

    std::unique_ptr<ICustomAllocator> GetDefaultAllocator() const override;
};

class MockLayerSupport : public LayerSupportBase
{
public:
    bool IsLayerSupported(const LayerType& type,
                          const std::vector<TensorInfo>& infos,
                          const BaseDescriptor& descriptor,
                          const Optional<LstmInputParamsInfo>& /*lstmParamsInfo*/,
                          const Optional<QuantizedLstmInputParamsInfo>& /*quantizedLstmParamsInfo*/,
                          Optional<std::string&> reasonIfUnsupported) const override
    {
        switch(type)
        {
            case LayerType::Input:
                return IsInputSupported(infos[0], reasonIfUnsupported);
            case LayerType::Output:
                return IsOutputSupported(infos[0], reasonIfUnsupported);
            case LayerType::Addition:
                return IsAdditionSupported(infos[0], infos[1], infos[2], reasonIfUnsupported);
            case LayerType::Convolution2d:
            {
                if (infos.size() != 4)
                {
                    throw InvalidArgumentException("Invalid number of TransposeConvolution2d "
                                                   "TensorInfos. TensorInfos should be of format: "
                                                   "{input, output, weights, biases}.");
                }

                auto desc = *(PolymorphicDowncast<const Convolution2dDescriptor*>(&descriptor));
                if (infos[3] == TensorInfo())
                {
                    return IsConvolution2dSupported(infos[0],
                                                    infos[1],
                                                    desc,
                                                    infos[2],
                                                    EmptyOptional(),
                                                    reasonIfUnsupported);
                }
                else
                {
                    return IsConvolution2dSupported(infos[0],
                                                    infos[1],
                                                    desc,
                                                    infos[2],
                                                    infos[3],
                                                    reasonIfUnsupported);
                }
            }
            default:
                return false;
        }
    }

    bool IsInputSupported(const TensorInfo& /*input*/,
                          Optional<std::string&> /*reasonIfUnsupported = EmptyOptional()*/) const override
    {
        return true;
    }

    bool IsOutputSupported(const TensorInfo& /*input*/,
                           Optional<std::string&> /*reasonIfUnsupported = EmptyOptional()*/) const override
    {
        return true;
    }

    bool IsAdditionSupported(const TensorInfo& /*input0*/,
                             const TensorInfo& /*input1*/,
                             const TensorInfo& /*output*/,
                             Optional<std::string&> /*reasonIfUnsupported = EmptyOptional()*/) const override
    {
        return true;
    }

    bool IsConvolution2dSupported(const TensorInfo& /*input*/,
                                  const TensorInfo& /*output*/,
                                  const Convolution2dDescriptor& /*descriptor*/,
                                  const TensorInfo& /*weights*/,
                                  const Optional<TensorInfo>& /*biases*/,
                                  Optional<std::string&> /*reasonIfUnsupported = EmptyOptional()*/) const override
    {
        return true;
    }
};

}    // namespace armnn
