//
// Copyright © 2022-2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <armnn/Deprecated.hpp>
#include <armnn/Descriptors.hpp>
#include <armnn/Exceptions.hpp>
#include <armnn/IRuntime.hpp>
#include <armnn/Optional.hpp>
#include <armnn/Tensor.hpp>
#include <armnn/Types.hpp>
#include <armnn/backends/IBackendInternal.hpp>
#include <armnn/backends/MemCopyWorkload.hpp>
#include <armnn/backends/ITensorHandle.hpp>
#include <armnn/backends/IWorkload.hpp>
#include <armnn/backends/OptimizationViews.hpp>
#include <armnn/backends/SubgraphView.hpp>
#include <armnn/backends/WorkloadData.hpp>
#include <armnn/backends/WorkloadFactory.hpp>
#include <armnn/backends/WorkloadInfo.hpp>
#include <armnn/utility/IgnoreUnused.hpp>
#include <armnn/utility/PolymorphicDowncast.hpp>
#include <armnnTestUtils/MockTensorHandle.hpp>
#include <backendsCommon/LayerSupportBase.hpp>

#include <client/include/CounterValue.hpp>
#include <client/include/ISendTimelinePacket.hpp>
#include <client/include/Timestamp.hpp>
#include <client/include/backends/IBackendProfiling.hpp>
#include <client/include/backends/IBackendProfilingContext.hpp>
#include <common/include/Optional.hpp>

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace armnn
{
class BackendId;
class ICustomAllocator;
class MockMemoryManager;
struct LstmInputParamsInfo;
struct QuantizedLstmInputParamsInfo;

// A bare bones Mock backend to enable unit testing of simple tensor manipulation features.
class MockBackend : public IBackendInternal
{
public:
    MockBackend() = default;

    ~MockBackend() = default;

    static const BackendId& GetIdStatic();

    const BackendId& GetId() const override
    {
        return GetIdStatic();
    }
    IBackendInternal::IWorkloadFactoryPtr
        CreateWorkloadFactory(const IBackendInternal::IMemoryManagerSharedPtr& memoryManager = nullptr) const override;

    IBackendInternal::ILayerSupportSharedPtr GetLayerSupport() const override;

    IBackendInternal::IMemoryManagerUniquePtr CreateMemoryManager() const override;

    IBackendInternal::IBackendContextPtr CreateBackendContext(const IRuntime::CreationOptions&) const override;
    IBackendInternal::IBackendProfilingContextPtr
    CreateBackendProfilingContext(const IRuntime::CreationOptions& creationOptions,
                                  IBackendProfilingPtr& backendProfiling) override;

    OptimizationViews OptimizeSubgraphView(const SubgraphView& subgraph) const override;

    std::unique_ptr<ICustomAllocator> GetDefaultAllocator() const override;
};

class MockWorkloadFactory : public IWorkloadFactory
{

public:
    explicit MockWorkloadFactory(const std::shared_ptr<MockMemoryManager>& memoryManager);
    MockWorkloadFactory();

    ~MockWorkloadFactory()
    {}

    const BackendId& GetBackendId() const override;

    bool SupportsSubTensors() const override
    {
        return false;
    }

    ARMNN_DEPRECATED_MSG("Use ITensorHandleFactory::CreateSubTensorHandle instead")
    std::unique_ptr<ITensorHandle> CreateSubTensorHandle(ITensorHandle&,
                                                         TensorShape const&,
                                                         unsigned int const*) const override
    {
        return nullptr;
    }

    ARMNN_DEPRECATED_MSG("Use ITensorHandleFactory::CreateTensorHandle instead")
    std::unique_ptr<ITensorHandle> CreateTensorHandle(const TensorInfo& tensorInfo,
                                                      const bool IsMemoryManaged = true) const override
    {
        IgnoreUnused(IsMemoryManaged);
        return std::make_unique<MockTensorHandle>(tensorInfo, m_MemoryManager);
    };

    ARMNN_DEPRECATED_MSG("Use ITensorHandleFactory::CreateTensorHandle instead")
    std::unique_ptr<ITensorHandle> CreateTensorHandle(const TensorInfo& tensorInfo,
                                                      DataLayout dataLayout,
                                                      const bool IsMemoryManaged = true) const override
    {
        IgnoreUnused(dataLayout, IsMemoryManaged);
        return std::make_unique<MockTensorHandle>(tensorInfo, static_cast<unsigned int>(MemorySource::Malloc));
    };

    ARMNN_DEPRECATED_MSG_REMOVAL_DATE(
        "Use ABI stable "
        "CreateWorkload(LayerType, const QueueDescriptor&, const WorkloadInfo& info) instead.",
        "23.08")
    std::unique_ptr<IWorkload> CreateInput(const InputQueueDescriptor& descriptor,
                                           const WorkloadInfo& info) const override
    {
        if (info.m_InputTensorInfos.empty())
        {
            throw InvalidArgumentException("MockWorkloadFactory::CreateInput: Input cannot be zero length");
        }
        if (info.m_OutputTensorInfos.empty())
        {
            throw InvalidArgumentException("MockWorkloadFactory::CreateInput: Output cannot be zero length");
        }

        if (info.m_InputTensorInfos[0].GetNumBytes() != info.m_OutputTensorInfos[0].GetNumBytes())
        {
            throw InvalidArgumentException(
                "MockWorkloadFactory::CreateInput: data input and output differ in byte count.");
        }

        return std::make_unique<CopyMemGenericWorkload>(descriptor, info);
    };

    std::unique_ptr<IWorkload>
        CreateWorkload(LayerType type, const QueueDescriptor& descriptor, const WorkloadInfo& info) const override;

private:
    mutable std::shared_ptr<MockMemoryManager> m_MemoryManager;
};

class MockBackendInitialiser
{
public:
    MockBackendInitialiser();
    ~MockBackendInitialiser();
};

class MockBackendProfilingContext : public arm::pipe::IBackendProfilingContext
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
        std::unique_ptr<arm::pipe::IRegisterBackendCounters> counterRegistrar =
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

    arm::pipe::Optional<std::string> ActivateCounters(uint32_t capturePeriod, const std::vector<uint16_t>& counterIds)
    {
        if (capturePeriod == 0 || counterIds.size() == 0)
        {
            m_ActiveCounters.clear();
        }
        else if (capturePeriod == 15939u)
        {
            return arm::pipe::Optional<std::string>("ActivateCounters example test error");
        }
        m_CapturePeriod  = capturePeriod;
        m_ActiveCounters = counterIds;
        return arm::pipe::Optional<std::string>();
    }

    std::vector<arm::pipe::Timestamp> ReportCounterValues()
    {
        std::vector<arm::pipe::CounterValue> counterValues;

        for (auto counterId : m_ActiveCounters)
        {
            counterValues.emplace_back(arm::pipe::CounterValue{ counterId, counterId + 1u });
        }

        uint64_t timestamp = m_CapturePeriod;
        return { arm::pipe::Timestamp{ timestamp, counterValues } };
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
            case LayerType::ElementwiseBinary:
            {
                auto elementwiseDesc = *(PolymorphicDowncast<const ElementwiseBinaryDescriptor*>(&descriptor));
                return (elementwiseDesc.m_Operation == BinaryOperation::Add);
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
