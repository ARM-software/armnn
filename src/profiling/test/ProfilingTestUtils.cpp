//
// Copyright © 2019 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ProfilingTestUtils.hpp"
#include "ProfilingUtils.hpp"

#include <armnn/Descriptors.hpp>
#include <armnn/utility/Assert.hpp>
#include <armnn/utility/NumericCast.hpp>

#include <Processes.hpp>
#include <ProfilingService.hpp>
#include <armnnUtils/Threads.hpp>

#include <common/include/LabelsAndEventClasses.hpp>

#include <TestUtils.hpp>

#include <doctest/doctest.h>

uint32_t GetStreamMetaDataPacketSize()
{
    uint32_t sizeUint32 = sizeof(uint32_t);
    uint32_t payloadSize = 0;
    payloadSize += armnn::numeric_cast<uint32_t>(GetSoftwareInfo().size()) + 1;
    payloadSize += armnn::numeric_cast<uint32_t>(GetHardwareVersion().size()) + 1;
    payloadSize += armnn::numeric_cast<uint32_t>(GetSoftwareVersion().size()) + 1;
    payloadSize += armnn::numeric_cast<uint32_t>(GetProcessName().size()) + 1;

    // Add packetVersionEntries
    payloadSize += 13 * 2 * sizeUint32;
    // Add packetVersionCountSize
    payloadSize += sizeUint32;

    uint32_t headerSize = 2 * sizeUint32;
    uint32_t bodySize = 10 * sizeUint32;

    return headerSize + bodySize + payloadSize;
}

std::vector<BackendId> GetSuitableBackendRegistered()
{
    std::vector<BackendId> suitableBackends;
    if (BackendRegistryInstance().IsBackendRegistered(GetComputeDeviceAsCString(armnn::Compute::CpuRef)))
    {
        suitableBackends.push_back(armnn::Compute::CpuRef);
    }
    if (BackendRegistryInstance().IsBackendRegistered(GetComputeDeviceAsCString(armnn::Compute::CpuAcc)))
    {
        suitableBackends.push_back(armnn::Compute::CpuAcc);
    }
    if (BackendRegistryInstance().IsBackendRegistered(GetComputeDeviceAsCString(armnn::Compute::GpuAcc)))
    {
        suitableBackends.push_back(armnn::Compute::GpuAcc);
    }
    return suitableBackends;
}

inline unsigned int OffsetToNextWord(unsigned int numberOfBytes)
{
    unsigned int uint32_t_size = sizeof(uint32_t);

    unsigned int remainder = numberOfBytes % uint32_t_size;
    if (remainder == 0)
    {
        return numberOfBytes;
    }

    return numberOfBytes + uint32_t_size - remainder;
}

void VerifyTimelineHeaderBinary(const unsigned char* readableData,
                                unsigned int& offset,
                                uint32_t packetDataLength)
{
    ARMNN_ASSERT(readableData);

    // Utils
    unsigned int uint32_t_size = sizeof(uint32_t);

    // Check the TimelineEventClassBinaryPacket header
    uint32_t timelineBinaryPacketHeaderWord0 = ReadUint32(readableData, offset);
    uint32_t timelineBinaryPacketFamily      = (timelineBinaryPacketHeaderWord0 >> 26) & 0x0000003F;
    uint32_t timelineBinaryPacketClass       = (timelineBinaryPacketHeaderWord0 >> 19) & 0x0000007F;
    uint32_t timelineBinaryPacketType        = (timelineBinaryPacketHeaderWord0 >> 16) & 0x00000007;
    uint32_t timelineBinaryPacketStreamId    = (timelineBinaryPacketHeaderWord0 >>  0) & 0x00000007;
    CHECK(timelineBinaryPacketFamily   == 1);
    CHECK(timelineBinaryPacketClass    == 0);
    CHECK(timelineBinaryPacketType     == 1);
    CHECK(timelineBinaryPacketStreamId == 0);
    offset += uint32_t_size;
    uint32_t timelineBinaryPacketHeaderWord1   = ReadUint32(readableData, offset);
    uint32_t timelineBinaryPacketSequenceNumber = (timelineBinaryPacketHeaderWord1 >> 24) & 0x00000001;
    uint32_t timelineBinaryPacketDataLength     = (timelineBinaryPacketHeaderWord1 >>  0) & 0x00FFFFFF;
    CHECK(timelineBinaryPacketSequenceNumber == 0);
    CHECK(timelineBinaryPacketDataLength     == packetDataLength);
    offset += uint32_t_size;
}

ProfilingGuid VerifyTimelineLabelBinaryPacketData(Optional<ProfilingGuid> guid,
                                                  const std::string& label,
                                                  const unsigned char* readableData,
                                                  unsigned int& offset)
{
    ARMNN_ASSERT(readableData);

    // Utils
    unsigned int uint32_t_size = sizeof(uint32_t);
    unsigned int uint64_t_size = sizeof(uint64_t);
    unsigned int label_size    = armnn::numeric_cast<unsigned int>(label.size());

    // Check the decl id
    uint32_t eventClassDeclId = ReadUint32(readableData, offset);
    CHECK(eventClassDeclId == 0);

    // Check the profiling GUID
    offset += uint32_t_size;
    uint64_t readProfilingGuid = ReadUint64(readableData, offset);
    if (guid.has_value())
    {
        CHECK(readProfilingGuid == guid.value());
    }
    else
    {
        armnn::profiling::ProfilingService profilingService;
        CHECK(readProfilingGuid == profilingService.GetStaticId(label));
    }

    // Check the SWTrace label
    offset += uint64_t_size;
    uint32_t swTraceLabelLength = ReadUint32(readableData, offset);
    CHECK(swTraceLabelLength == label_size + 1);               // Label length including the null-terminator
    offset += uint32_t_size;
    CHECK(std::memcmp(readableData + offset,                  // Offset to the label in the buffer
                               label.data(),                           // The original label
                               swTraceLabelLength - 1) == 0);          // The length of the label

    // SWTrace strings are written in blocks of words, so the offset has to be updated to the next whole word
    offset += OffsetToNextWord(swTraceLabelLength);

    ProfilingGuid labelGuid(readProfilingGuid);
    return labelGuid;
}

void VerifyTimelineEventClassBinaryPacketData(ProfilingGuid guid,
                                              ProfilingGuid nameGuid,
                                              const unsigned char* readableData,
                                              unsigned int& offset)
{
    ARMNN_ASSERT(readableData);

    // Utils
    unsigned int uint32_t_size = sizeof(uint32_t);
    unsigned int uint64_t_size = sizeof(uint64_t);

    // Check the decl id
    uint32_t eventClassDeclId = ReadUint32(readableData, offset);
    CHECK(eventClassDeclId == 2);

    // Check the profiling GUID
    offset += uint32_t_size;
    uint64_t readProfilingGuid = ReadUint64(readableData, offset);
    CHECK(readProfilingGuid == guid);

    offset += uint64_t_size;
    uint64_t readProfiilngNameGuid = ReadUint64(readableData, offset);
    CHECK(readProfiilngNameGuid == nameGuid);

    // Update the offset to allow parsing to be continued after this function returns
    offset += uint64_t_size;
}

void VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType relationshipType,
                                                Optional<ProfilingGuid> relationshipGuid,
                                                Optional<ProfilingGuid> headGuid,
                                                Optional<ProfilingGuid> tailGuid,
                                                Optional<ProfilingGuid> attributeGuid,
                                                const unsigned char* readableData,
                                                unsigned int& offset)
{
    ARMNN_ASSERT(readableData);

    uint32_t relationshipTypeUint = 0;
    switch (relationshipType)
    {
        case ProfilingRelationshipType::RetentionLink:
            relationshipTypeUint = 0;
            break;
        case ProfilingRelationshipType::ExecutionLink:
            relationshipTypeUint = 1;
            break;
        case ProfilingRelationshipType::DataLink:
            relationshipTypeUint = 2;
            break;
        case ProfilingRelationshipType::LabelLink:
            relationshipTypeUint = 3;
            break;
        default:
            FAIL("Unknown relationship type");
    }

    // Utils
    unsigned int uint32_t_size = sizeof(uint32_t);
    unsigned int uint64_t_size = sizeof(uint64_t);

    // Check the decl id
    uint32_t eventClassDeclId = ReadUint32(readableData, offset);
    CHECK(eventClassDeclId == 3);

    // Check the relationship type
    offset += uint32_t_size;
    uint32_t readRelationshipTypeUint = ReadUint32(readableData, offset);
    CHECK(readRelationshipTypeUint == relationshipTypeUint);

    // Check the relationship GUID
    offset += uint32_t_size;
    uint64_t readRelationshipGuid = ReadUint64(readableData, offset);
    if (relationshipGuid.has_value())
    {
        CHECK(readRelationshipGuid == relationshipGuid.value());
    }
    else
    {
        CHECK(readRelationshipGuid != ProfilingGuid(0));
    }

    // Check the head GUID of the relationship
    offset += uint64_t_size;
    uint64_t readHeadRelationshipGuid = ReadUint64(readableData, offset);
    if (headGuid.has_value())
    {
        CHECK(readHeadRelationshipGuid == headGuid.value());
    }
    else
    {
        CHECK(readHeadRelationshipGuid != ProfilingGuid(0));
    }

    // Check the tail GUID of the relationship
    offset += uint64_t_size;
    uint64_t readTailRelationshipGuid = ReadUint64(readableData, offset);
    if (tailGuid.has_value())
    {
        CHECK(readTailRelationshipGuid == tailGuid.value());
    }
    else
    {
        CHECK(readTailRelationshipGuid != ProfilingGuid(0));
    }

    // Check the attribute GUID of the relationship
    offset += uint64_t_size;
    uint64_t readAttributeRelationshipGuid = ReadUint64(readableData, offset);
    if (attributeGuid.has_value())
    {
        CHECK(readAttributeRelationshipGuid == attributeGuid.value());
    }
    else
    {
        CHECK(readAttributeRelationshipGuid == ProfilingGuid(0));
    }

    // Update the offset to allow parsing to be continued after this function returns
    offset += uint64_t_size;
}

ProfilingGuid VerifyTimelineEntityBinaryPacketData(Optional<ProfilingGuid> guid,
                                                   const unsigned char* readableData,
                                                   unsigned int& offset)
{
    ARMNN_ASSERT(readableData);

    // Utils
    unsigned int uint32_t_size = sizeof(uint32_t);
    unsigned int uint64_t_size = sizeof(uint64_t);

    // Reading TimelineEntityClassBinaryPacket
    // Check the decl_id
    uint32_t entityDeclId = ReadUint32(readableData, offset);
    CHECK(entityDeclId == 1);

    // Check the profiling GUID
    offset += uint32_t_size;
    uint64_t readProfilingGuid = ReadUint64(readableData, offset);

    if (guid.has_value())
    {
        CHECK(readProfilingGuid == guid.value());
    }
    else
    {
        CHECK(readProfilingGuid != ProfilingGuid(0));
    }

    offset += uint64_t_size;

    ProfilingGuid entityGuid(readProfilingGuid);
    return entityGuid;
}

ProfilingGuid VerifyTimelineEventBinaryPacket(Optional<uint64_t> timestamp,
                                              Optional<int> threadId,
                                              Optional<ProfilingGuid> eventGuid,
                                              const unsigned char* readableData,
                                              unsigned int& offset)
{
    ARMNN_ASSERT(readableData);

    // Utils
    unsigned int uint32_t_size = sizeof(uint32_t);
    unsigned int uint64_t_size = sizeof(uint64_t);

    // Reading TimelineEventBinaryPacket
    // Check the decl_id
    uint32_t entityDeclId = ReadUint32(readableData, offset);
    CHECK(entityDeclId == 4);

    // Check the timestamp
    offset += uint32_t_size;
    uint64_t readTimestamp = ReadUint64(readableData, offset);
    if (timestamp.has_value())
    {
        CHECK(readTimestamp == timestamp.value());
    }
    else
    {
        CHECK(readTimestamp != 0);
    }

    // Check the thread id
    offset += uint64_t_size;
    std::vector<uint8_t> readThreadId(ThreadIdSize, 0);
    ReadBytes(readableData, offset, ThreadIdSize, readThreadId.data());
    if (threadId.has_value())
    {
        CHECK(readThreadId == threadId.value());
    }
    else
    {
        CHECK(readThreadId == armnnUtils::Threads::GetCurrentThreadId());
    }

    // Check the event GUID
    offset += ThreadIdSize;
    uint64_t readEventGuid = ReadUint64(readableData, offset);
    if (eventGuid.has_value())
    {
        CHECK(readEventGuid == eventGuid.value());
    }
    else
    {
        CHECK(readEventGuid != ProfilingGuid(0));
    }

    offset += uint64_t_size;

    ProfilingGuid eventid(readEventGuid);
    return eventid;
}

void VerifyPostOptimisationStructureTestImpl(armnn::BackendId backendId)
{
    using namespace armnn;

    // Create runtime in which test will run
    armnn::IRuntime::CreationOptions options;
    options.m_ProfilingOptions.m_EnableProfiling = true;
    options.m_ProfilingOptions.m_TimelineEnabled = true;
    armnn::RuntimeImpl runtime(options);
    GetProfilingService(&runtime).ResetExternalProfilingOptions(options.m_ProfilingOptions, false);

    profiling::ProfilingServiceRuntimeHelper profilingServiceHelper(GetProfilingService(&runtime));
    profilingServiceHelper.ForceTransitionToState(ProfilingState::NotConnected);
    profilingServiceHelper.ForceTransitionToState(ProfilingState::WaitingForAck);
    profilingServiceHelper.ForceTransitionToState(ProfilingState::Active);

    // build up the structure of the network
    INetworkPtr net(INetwork::Create());

    // Convolution details
    TensorInfo inputInfo({ 1, 2, 5, 1 }, DataType::Float32);
    TensorInfo weightInfo({ 3, 2, 3, 1 }, DataType::Float32, 0.0f, 0, true);
    TensorInfo biasInfo({ 3 }, DataType::Float32, 0.0f, 0, true);
    TensorInfo outputInfo({ 1, 3, 7, 1 }, DataType::Float32);
    std::vector<float> weightsData{
        1.0f, 0.0f, 0.0f,
        0.0f, 2.0f, -1.5f,

        0.0f, 0.0f, 0.0f,
        0.2f, 0.2f, 0.2f,

        0.5f, 0.0f, 0.5f,
        0.0f, -1.0f, 0.0f
    };
    ConstTensor weights(weightInfo, weightsData);

    Optional<ConstTensor> optionalBiases;
    std::vector<float> biasesData{ 1.0f, 0.0f, 0.0f };
    ConstTensor biases(biasInfo, biasesData);
    optionalBiases = Optional<ConstTensor>(biases);

    // Input layer
    IConnectableLayer* input = net->AddInputLayer(0, "input");

    // Convolution2d layer
    Convolution2dDescriptor conv2dDesc;
    conv2dDesc.m_StrideX = 1;
    conv2dDesc.m_StrideY = 1;
    conv2dDesc.m_PadLeft = 0;
    conv2dDesc.m_PadRight = 0;
    conv2dDesc.m_PadTop = 2;
    conv2dDesc.m_PadBottom = 2;
    conv2dDesc.m_BiasEnabled = true;
    IConnectableLayer* conv2d = net->AddConvolution2dLayer(conv2dDesc, weights, optionalBiases);

    // Abs layer
    armnn::ElementwiseUnaryDescriptor absDesc;
    armnn::IConnectableLayer* const abs = net->AddElementwiseUnaryLayer(absDesc, "abs");

    // Output layer
    IConnectableLayer* output = net->AddOutputLayer(0, "output");

    input->GetOutputSlot(0).Connect(conv2d->GetInputSlot(0));
    conv2d->GetOutputSlot(0).Connect(abs->GetInputSlot(0));
    abs->GetOutputSlot(0).Connect(output->GetInputSlot(0));

    input->GetOutputSlot(0).SetTensorInfo(inputInfo);
    conv2d->GetOutputSlot(0).SetTensorInfo(outputInfo);
    abs->GetOutputSlot(0).SetTensorInfo(outputInfo);

    // optimize the network
    std::vector<armnn::BackendId> backends = { backendId };
    IOptimizedNetworkPtr optNet = Optimize(*net, backends, runtime.GetDeviceSpec());

    ProfilingGuid optNetGuid = optNet->GetGuid();

    // Load it into the runtime. It should success.
    armnn::NetworkId netId;
    CHECK(runtime.LoadNetwork(netId, std::move(optNet)) == Status::Success);

    profiling::BufferManager& bufferManager = profilingServiceHelper.GetProfilingBufferManager();
    auto readableBuffer = bufferManager.GetReadableBuffer();

    // Profiling is enabled, the post-optimisation structure should be created
    CHECK(readableBuffer != nullptr);
    unsigned int size = readableBuffer->GetSize();

    const unsigned char* readableData = readableBuffer->GetReadableData();
    CHECK(readableData != nullptr);

    unsigned int offset = 0;

    // Verify Header
    VerifyTimelineHeaderBinary(readableData, offset, size - 8);

    // Post-optimisation network
    // Network entity
    VerifyTimelineEntityBinaryPacketData(optNetGuid, readableData, offset);

    // Entity - Type relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::LabelLink,
                                               EmptyOptional(),
                                               optNetGuid,
                                               LabelsAndEventClasses::NETWORK_GUID,
                                               LabelsAndEventClasses::TYPE_GUID,
                                               readableData,
                                               offset);

    // Network - START OF LIFE
    ProfilingGuid networkSolEventGuid = VerifyTimelineEventBinaryPacket(EmptyOptional(),
                                                                        EmptyOptional(),
                                                                        EmptyOptional(),
                                                                        readableData,
                                                                        offset);

    // Network - START OF LIFE event relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::ExecutionLink,
                                               EmptyOptional(),
                                               optNetGuid,
                                               networkSolEventGuid,
                                               LabelsAndEventClasses::ARMNN_PROFILING_SOL_EVENT_CLASS,
                                               readableData,
                                               offset);

    // Process ID Label
    int processID = armnnUtils::Processes::GetCurrentId();
    std::stringstream ss;
    ss << processID;
    std::string processIdLabel = ss.str();
    VerifyTimelineLabelBinaryPacketData(EmptyOptional(), processIdLabel, readableData, offset);

    // Entity - Process ID relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::LabelLink,
                                               EmptyOptional(),
                                               optNetGuid,
                                               EmptyOptional(),
                                               LabelsAndEventClasses::PROCESS_ID_GUID,
                                               readableData,
                                               offset);

    // Input layer
    // Input layer entity
    VerifyTimelineEntityBinaryPacketData(input->GetGuid(), readableData, offset);

    // Name Entity
    ProfilingGuid inputLabelGuid = VerifyTimelineLabelBinaryPacketData(EmptyOptional(), "input", readableData, offset);

    // Entity - Name relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::LabelLink,
                                               EmptyOptional(),
                                               input->GetGuid(),
                                               inputLabelGuid,
                                               LabelsAndEventClasses::NAME_GUID,
                                               readableData,
                                               offset);

    // Entity - Type relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::LabelLink,
                                               EmptyOptional(),
                                               input->GetGuid(),
                                               LabelsAndEventClasses::LAYER_GUID,
                                               LabelsAndEventClasses::TYPE_GUID,
                                               readableData,
                                               offset);

    // Network - Input layer relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::RetentionLink,
                                               EmptyOptional(),
                                               optNetGuid,
                                               input->GetGuid(),
                                               LabelsAndEventClasses::CHILD_GUID,
                                               readableData,
                                               offset);

    // Conv2d layer
    // Conv2d layer entity
    VerifyTimelineEntityBinaryPacketData(conv2d->GetGuid(), readableData, offset);

    // Name entity
    ProfilingGuid conv2dNameLabelGuid = VerifyTimelineLabelBinaryPacketData(
        EmptyOptional(), "<Unnamed>", readableData, offset);

    // Entity - Name relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::LabelLink,
                                               EmptyOptional(),
                                               conv2d->GetGuid(),
                                               conv2dNameLabelGuid,
                                               LabelsAndEventClasses::NAME_GUID,
                                               readableData,
                                               offset);

    // Entity - Type relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::LabelLink,
                                               EmptyOptional(),
                                               conv2d->GetGuid(),
                                               LabelsAndEventClasses::LAYER_GUID,
                                               LabelsAndEventClasses::TYPE_GUID,
                                               readableData,
                                               offset);

    // Network - Conv2d layer relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::RetentionLink,
                                               EmptyOptional(),
                                               optNetGuid,
                                               conv2d->GetGuid(),
                                               LabelsAndEventClasses::CHILD_GUID,
                                               readableData,
                                               offset);

    // Input layer - Conv2d layer relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::RetentionLink,
                                               EmptyOptional(),
                                               input->GetGuid(),
                                               conv2d->GetGuid(),
                                               LabelsAndEventClasses::CONNECTION_GUID,
                                               readableData,
                                               offset);

    // Conv2d workload
    // Conv2d workload entity
    ProfilingGuid conv2DWorkloadGuid = VerifyTimelineEntityBinaryPacketData(EmptyOptional(), readableData, offset);

    // Entity - Type relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::LabelLink,
                                               EmptyOptional(),
                                               conv2DWorkloadGuid,
                                               LabelsAndEventClasses::WORKLOAD_GUID,
                                               LabelsAndEventClasses::TYPE_GUID,
                                               readableData,
                                               offset);

    // BackendId entity
    ProfilingGuid backendIdLabelGuid = VerifyTimelineLabelBinaryPacketData(
        EmptyOptional(), backendId.Get(), readableData, offset);

    // Entity - BackendId relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::LabelLink,
                                               EmptyOptional(),
                                               conv2DWorkloadGuid,
                                               backendIdLabelGuid,
                                               LabelsAndEventClasses::BACKENDID_GUID,
                                               readableData,
                                               offset);


    // Conv2d layer - Conv2d workload relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::RetentionLink,
                                               EmptyOptional(),
                                               conv2d->GetGuid(),
                                               conv2DWorkloadGuid,
                                               LabelsAndEventClasses::CHILD_GUID,
                                               readableData,
                                               offset);

    // Abs layer
    // Abs layer entity
    VerifyTimelineEntityBinaryPacketData(abs->GetGuid(), readableData, offset);

    // Name entity
    ProfilingGuid absLabelGuid = VerifyTimelineLabelBinaryPacketData(
        EmptyOptional(), "abs", readableData, offset);

    // Entity - Name relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::LabelLink,
                                               EmptyOptional(),
                                               abs->GetGuid(),
                                               absLabelGuid,
                                               LabelsAndEventClasses::NAME_GUID,
                                               readableData,
                                               offset);

    // Entity - Type relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::LabelLink,
                                               EmptyOptional(),
                                               abs->GetGuid(),
                                               LabelsAndEventClasses::LAYER_GUID,
                                               LabelsAndEventClasses::TYPE_GUID,
                                               readableData,
                                               offset);

    // Network - Abs layer relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::RetentionLink,
                                               EmptyOptional(),
                                               optNetGuid,
                                               abs->GetGuid(),
                                               LabelsAndEventClasses::CHILD_GUID,
                                               readableData,
                                               offset);

    // Conv2d layer - Abs layer relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::RetentionLink,
                                               EmptyOptional(),
                                               conv2d->GetGuid(),
                                               abs->GetGuid(),
                                               LabelsAndEventClasses::CONNECTION_GUID,
                                               readableData,
                                               offset);

    // Abs workload
    // Abs workload entity
    ProfilingGuid absWorkloadGuid = VerifyTimelineEntityBinaryPacketData(EmptyOptional(), readableData, offset);

    // Entity - Type relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::LabelLink,
                                               EmptyOptional(),
                                               absWorkloadGuid,
                                               LabelsAndEventClasses::WORKLOAD_GUID,
                                               LabelsAndEventClasses::TYPE_GUID,
                                               readableData,
                                               offset);

    // BackendId entity
    VerifyTimelineLabelBinaryPacketData(EmptyOptional(), backendId.Get(), readableData, offset);

    // Entity - BackendId relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::LabelLink,
                                               EmptyOptional(),
                                               absWorkloadGuid,
                                               backendIdLabelGuid,
                                               LabelsAndEventClasses::BACKENDID_GUID,
                                               readableData,
                                               offset);

    // Abs layer - Abs workload relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::RetentionLink,
                                               EmptyOptional(),
                                               abs->GetGuid(),
                                               absWorkloadGuid,
                                               LabelsAndEventClasses::CHILD_GUID,
                                               readableData,
                                               offset);

    // Output layer
    // Output layer entity
    VerifyTimelineEntityBinaryPacketData(output->GetGuid(), readableData, offset);

    // Name entity
    ProfilingGuid outputLabelGuid = VerifyTimelineLabelBinaryPacketData(
        EmptyOptional(), "output", readableData, offset);

    // Entity - Name relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::LabelLink,
                                               EmptyOptional(),
                                               output->GetGuid(),
                                               outputLabelGuid,
                                               LabelsAndEventClasses::NAME_GUID,
                                               readableData,
                                               offset);

    // Entity - Type relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::LabelLink,
                                               EmptyOptional(),
                                               output->GetGuid(),
                                               LabelsAndEventClasses::LAYER_GUID,
                                               LabelsAndEventClasses::TYPE_GUID,
                                               readableData,
                                               offset);

    // Network - Output layer relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::RetentionLink,
                                               EmptyOptional(),
                                               optNetGuid,
                                               output->GetGuid(),
                                               LabelsAndEventClasses::CHILD_GUID,
                                               readableData,
                                               offset);

    // Abs layer - Output layer relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::RetentionLink,
                                               EmptyOptional(),
                                               abs->GetGuid(),
                                               output->GetGuid(),
                                               LabelsAndEventClasses::CONNECTION_GUID,
                                               readableData,
                                               offset);

    bufferManager.MarkRead(readableBuffer);

    // Creates structures for input & output.
    std::vector<float> inputData(inputInfo.GetNumElements());
    std::vector<float> outputData(outputInfo.GetNumElements());

    TensorInfo inputTensorInfo = runtime.GetInputTensorInfo(netId, 0);
    inputTensorInfo.SetConstant(true);
    InputTensors inputTensors
        {
        {0, ConstTensor(inputTensorInfo, inputData.data())}
        };
    OutputTensors outputTensors
        {
        {0, Tensor(runtime.GetOutputTensorInfo(netId, 0), outputData.data())}
        };

    // Does the inference.
    runtime.EnqueueWorkload(netId, inputTensors, outputTensors);

    // Get readable buffer for input workload
    auto inputReadableBuffer = bufferManager.GetReadableBuffer();
    CHECK(inputReadableBuffer != nullptr);

    // Get readable buffer for output workload
    auto outputReadableBuffer = bufferManager.GetReadableBuffer();
    CHECK(outputReadableBuffer != nullptr);

    // Get readable buffer for inference timeline
    auto inferenceReadableBuffer = bufferManager.GetReadableBuffer();
    CHECK(inferenceReadableBuffer != nullptr);

    // Validate input workload data
    size = inputReadableBuffer->GetSize();
    CHECK(size == 164);

    readableData = inputReadableBuffer->GetReadableData();
    CHECK(readableData != nullptr);

    offset = 0;

    // Verify Header
    VerifyTimelineHeaderBinary(readableData, offset, 156);

    // Input workload
    // Input workload entity
    ProfilingGuid inputWorkloadGuid = VerifyTimelineEntityBinaryPacketData(EmptyOptional(), readableData, offset);

    // Entity - Type relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::LabelLink,
                                               EmptyOptional(),
                                               inputWorkloadGuid,
                                               LabelsAndEventClasses::WORKLOAD_GUID,
                                               LabelsAndEventClasses::TYPE_GUID,
                                               readableData,
                                               offset);

    // BackendId entity
    VerifyTimelineLabelBinaryPacketData(EmptyOptional(), backendId.Get(), readableData, offset);

    // Entity - BackendId relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::LabelLink,
                                               EmptyOptional(),
                                               inputWorkloadGuid,
                                               backendIdLabelGuid,
                                               LabelsAndEventClasses::BACKENDID_GUID,
                                               readableData,
                                               offset);

    // Input layer - Input workload relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::RetentionLink,
                                               EmptyOptional(),
                                               input->GetGuid(),
                                               inputWorkloadGuid,
                                               LabelsAndEventClasses::CHILD_GUID,
                                               readableData,
                                               offset);

    bufferManager.MarkRead(inputReadableBuffer);

    // Validate output workload data
    size = outputReadableBuffer->GetSize();
    CHECK(size == 164);

    readableData = outputReadableBuffer->GetReadableData();
    CHECK(readableData != nullptr);

    offset = 0;

    // Verify Header
    VerifyTimelineHeaderBinary(readableData, offset, 156);

    // Output workload
    // Output workload entity
    ProfilingGuid outputWorkloadGuid = VerifyTimelineEntityBinaryPacketData(EmptyOptional(), readableData, offset);

    // Entity - Type relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::LabelLink,
                                               EmptyOptional(),
                                               outputWorkloadGuid,
                                               LabelsAndEventClasses::WORKLOAD_GUID,
                                               LabelsAndEventClasses::TYPE_GUID,
                                               readableData,
                                               offset);

    // BackendId entity
    VerifyTimelineLabelBinaryPacketData(EmptyOptional(), backendId.Get(), readableData, offset);

    // Entity - BackendId relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::LabelLink,
                                               EmptyOptional(),
                                               outputWorkloadGuid,
                                               backendIdLabelGuid,
                                               LabelsAndEventClasses::BACKENDID_GUID,
                                               readableData,
                                               offset);

    // Output layer - Output workload relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::RetentionLink,
                                               EmptyOptional(),
                                               output->GetGuid(),
                                               outputWorkloadGuid,
                                               LabelsAndEventClasses::CHILD_GUID,
                                               readableData,
                                               offset);

    bufferManager.MarkRead(outputReadableBuffer);

    // Validate inference data
    size = inferenceReadableBuffer->GetSize();

    CHECK(size == 1228 + 10 * ThreadIdSize);

    readableData = inferenceReadableBuffer->GetReadableData();
    CHECK(readableData != nullptr);

    offset = 0;

    // Verify Header
    VerifyTimelineHeaderBinary(readableData, offset, 1220 + 10 * ThreadIdSize);

    // Inference timeline trace
    // Inference entity
    ProfilingGuid inferenceGuid = VerifyTimelineEntityBinaryPacketData(EmptyOptional(), readableData, offset);

    // Entity - Type relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::LabelLink,
                                               EmptyOptional(),
                                               inferenceGuid,
                                               LabelsAndEventClasses::INFERENCE_GUID,
                                               LabelsAndEventClasses::TYPE_GUID,
                                               readableData,
                                               offset);

    // Network - Inference relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::RetentionLink,
                                               EmptyOptional(),
                                               optNetGuid,
                                               inferenceGuid,
                                               LabelsAndEventClasses::EXECUTION_OF_GUID,
                                               readableData,
                                               offset);

    // Start Inference life
    // Event packet - timeline, threadId, eventGuid
    ProfilingGuid inferenceEventGuid = VerifyTimelineEventBinaryPacket(
        EmptyOptional(), EmptyOptional(), EmptyOptional(), readableData, offset);

    // Inference - event relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::ExecutionLink,
                                               EmptyOptional(),
                                               inferenceGuid,
                                               inferenceEventGuid,
                                               LabelsAndEventClasses::ARMNN_PROFILING_SOL_EVENT_CLASS,
                                               readableData,
                                               offset);

    // Execution
    // Input workload execution
    // Input workload execution entity
    ProfilingGuid inputWorkloadExecutionGuid = VerifyTimelineEntityBinaryPacketData(
        EmptyOptional(), readableData, offset);

    // Entity - Type relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::LabelLink,
                                               EmptyOptional(),
                                               inputWorkloadExecutionGuid,
                                               LabelsAndEventClasses::WORKLOAD_EXECUTION_GUID,
                                               LabelsAndEventClasses::TYPE_GUID,
                                               readableData,
                                               offset);

    // Inference - Workload execution relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::RetentionLink,
                                               EmptyOptional(),
                                               inferenceGuid,
                                               inputWorkloadExecutionGuid,
                                               LabelsAndEventClasses::CHILD_GUID,
                                               readableData,
                                               offset);

    // Workload - Workload execution relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::RetentionLink,
                                               EmptyOptional(),
                                               inputWorkloadGuid,
                                               inputWorkloadExecutionGuid,
                                               LabelsAndEventClasses::EXECUTION_OF_GUID,
                                               readableData,
                                               offset);

    // Start Input workload execution life
    // Event packet - timeline, threadId, eventGuid
    ProfilingGuid inputWorkloadExecutionSOLEventId = VerifyTimelineEventBinaryPacket(
        EmptyOptional(), EmptyOptional(), EmptyOptional(), readableData, offset);

    // Input workload execution - event relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::ExecutionLink,
                                               EmptyOptional(),
                                               inputWorkloadExecutionGuid,
                                               inputWorkloadExecutionSOLEventId,
                                               LabelsAndEventClasses::ARMNN_PROFILING_SOL_EVENT_CLASS,
                                               readableData,
                                               offset);

    // End of Input workload execution life
    // Event packet - timeline, threadId, eventGuid
    ProfilingGuid inputWorkloadExecutionEOLEventId = VerifyTimelineEventBinaryPacket(
        EmptyOptional(), EmptyOptional(), EmptyOptional(), readableData, offset);

    // Input workload execution - event relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::ExecutionLink,
                                               EmptyOptional(),
                                               inputWorkloadExecutionGuid,
                                               inputWorkloadExecutionEOLEventId,
                                               LabelsAndEventClasses::ARMNN_PROFILING_EOL_EVENT_CLASS,
                                               readableData,
                                               offset);

    // Conv2d workload execution
    // Conv2d workload execution entity
    ProfilingGuid conv2DWorkloadExecutionGuid = VerifyTimelineEntityBinaryPacketData(
        EmptyOptional(), readableData, offset);

    // Entity - Type relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::LabelLink,
                                               EmptyOptional(),
                                               conv2DWorkloadExecutionGuid,
                                               LabelsAndEventClasses::WORKLOAD_EXECUTION_GUID,
                                               LabelsAndEventClasses::TYPE_GUID,
                                               readableData,
                                               offset);

    // Inference - Workload execution relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::RetentionLink,
                                               EmptyOptional(),
                                               inferenceGuid,
                                               conv2DWorkloadExecutionGuid,
                                               LabelsAndEventClasses::CHILD_GUID,
                                               readableData,
                                               offset);

    // Workload - Workload execution relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::RetentionLink,
                                               EmptyOptional(),
                                               conv2DWorkloadGuid,
                                               conv2DWorkloadExecutionGuid,
                                               LabelsAndEventClasses::EXECUTION_OF_GUID,
                                               readableData,
                                               offset);

    // Start Conv2d workload execution life
    // Event packet - timeline, threadId, eventGuid
    ProfilingGuid conv2DWorkloadExecutionSOLEventGuid = VerifyTimelineEventBinaryPacket(
        EmptyOptional(), EmptyOptional(), EmptyOptional(), readableData, offset);

    // Conv2d workload execution - event relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::ExecutionLink,
                                               EmptyOptional(),
                                               conv2DWorkloadExecutionGuid,
                                               conv2DWorkloadExecutionSOLEventGuid,
                                               LabelsAndEventClasses::ARMNN_PROFILING_SOL_EVENT_CLASS,
                                               readableData,
                                               offset);

    // End of Conv2d workload execution life
    // Event packet - timeline, threadId, eventGuid
    ProfilingGuid conv2DWorkloadExecutionEOLEventGuid = VerifyTimelineEventBinaryPacket(
        EmptyOptional(), EmptyOptional(), EmptyOptional(), readableData, offset);

    // Conv2d workload execution - event relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::ExecutionLink,
                                               EmptyOptional(),
                                               conv2DWorkloadExecutionGuid,
                                               conv2DWorkloadExecutionEOLEventGuid,
                                               LabelsAndEventClasses::ARMNN_PROFILING_EOL_EVENT_CLASS,
                                               readableData,
                                               offset);

    // Abs workload execution
    // Abs workload execution entity
    ProfilingGuid absWorkloadExecutionGuid = VerifyTimelineEntityBinaryPacketData(
        EmptyOptional(), readableData, offset);

    // Entity - Type relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::LabelLink,
                                               EmptyOptional(),
                                               absWorkloadExecutionGuid,
                                               LabelsAndEventClasses::WORKLOAD_EXECUTION_GUID,
                                               LabelsAndEventClasses::TYPE_GUID,
                                               readableData,
                                               offset);

    // Inference - Workload execution relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::RetentionLink,
                                               EmptyOptional(),
                                               inferenceGuid,
                                               absWorkloadExecutionGuid,
                                               LabelsAndEventClasses::CHILD_GUID,
                                               readableData,
                                               offset);

    // Workload - Workload execution relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::RetentionLink,
                                               EmptyOptional(),
                                               absWorkloadGuid,
                                               absWorkloadExecutionGuid,
                                               LabelsAndEventClasses::EXECUTION_OF_GUID,
                                               readableData,
                                               offset);

    // Start Abs workload execution life
    // Event packet - timeline, threadId, eventGuid
    ProfilingGuid absWorkloadExecutionSOLEventGuid = VerifyTimelineEventBinaryPacket(
        EmptyOptional(), EmptyOptional(), EmptyOptional(), readableData, offset);

    // Abs workload execution - event relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::ExecutionLink,
                                               EmptyOptional(),
                                               absWorkloadExecutionGuid,
                                               absWorkloadExecutionSOLEventGuid,
                                               LabelsAndEventClasses::ARMNN_PROFILING_SOL_EVENT_CLASS,
                                               readableData,
                                               offset);

    // End of Abs workload execution life
    // Event packet - timeline, threadId, eventGuid
    ProfilingGuid absWorkloadExecutionEOLEventGuid = VerifyTimelineEventBinaryPacket(
        EmptyOptional(), EmptyOptional(), EmptyOptional(), readableData, offset);

    // Abs workload execution - event relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::ExecutionLink,
                                               EmptyOptional(),
                                               absWorkloadExecutionGuid,
                                               absWorkloadExecutionEOLEventGuid,
                                               LabelsAndEventClasses::ARMNN_PROFILING_EOL_EVENT_CLASS,
                                               readableData,
                                               offset);

    // Output workload execution
    // Output workload execution entity
    ProfilingGuid outputWorkloadExecutionGuid = VerifyTimelineEntityBinaryPacketData(
        EmptyOptional(), readableData, offset);

    // Entity - Type relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::LabelLink,
                                               EmptyOptional(),
                                               outputWorkloadExecutionGuid,
                                               LabelsAndEventClasses::WORKLOAD_EXECUTION_GUID,
                                               LabelsAndEventClasses::TYPE_GUID,
                                               readableData,
                                               offset);

    // Inference - Workload execution relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::RetentionLink,
                                               EmptyOptional(),
                                               inferenceGuid,
                                               outputWorkloadExecutionGuid,
                                               LabelsAndEventClasses::CHILD_GUID,
                                               readableData,
                                               offset);

    // Workload - Workload execution relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::RetentionLink,
                                               EmptyOptional(),
                                               outputWorkloadGuid,
                                               outputWorkloadExecutionGuid,
                                               LabelsAndEventClasses::EXECUTION_OF_GUID,
                                               readableData,
                                               offset);

    // Start Output workload execution life
    // Event packet - timeline, threadId, eventGuid
    ProfilingGuid outputWorkloadExecutionSOLEventGuid = VerifyTimelineEventBinaryPacket(
        EmptyOptional(), EmptyOptional(), EmptyOptional(), readableData, offset);

    // Output workload execution - event relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::ExecutionLink,
                                               EmptyOptional(),
                                               outputWorkloadExecutionGuid,
                                               outputWorkloadExecutionSOLEventGuid,
                                               LabelsAndEventClasses::ARMNN_PROFILING_SOL_EVENT_CLASS,
                                               readableData,
                                               offset);

    // End of Normalize workload execution life
    // Event packet - timeline, threadId, eventGuid
    ProfilingGuid outputWorkloadExecutionEOLEventGuid = VerifyTimelineEventBinaryPacket(
        EmptyOptional(), EmptyOptional(), EmptyOptional(), readableData, offset);

    // Output workload execution - event relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::ExecutionLink,
                                               EmptyOptional(),
                                               outputWorkloadExecutionGuid,
                                               outputWorkloadExecutionEOLEventGuid,
                                               LabelsAndEventClasses::ARMNN_PROFILING_EOL_EVENT_CLASS,
                                               readableData,
                                               offset);

    // End of Inference life
    // Event packet - timeline, threadId, eventGuid
    ProfilingGuid inferenceEOLEventGuid = VerifyTimelineEventBinaryPacket(
        EmptyOptional(), EmptyOptional(), EmptyOptional(), readableData, offset);

    // Inference - event relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::ExecutionLink,
                                               EmptyOptional(),
                                               inferenceGuid,
                                               inferenceEOLEventGuid,
                                               LabelsAndEventClasses::ARMNN_PROFILING_EOL_EVENT_CLASS,
                                               readableData,
                                               offset);

    bufferManager.MarkRead(inferenceReadableBuffer);
}

bool CompareOutput(std::vector<std::string> output, std::vector<std::string> expectedOutput)
{
    if (output.size() != expectedOutput.size())
    {
        std::cerr << "output has [" << output.size() << "] lines, expected was ["
                  << expectedOutput.size() << "] lines" << std::endl;
        std::cerr << std::endl << "actual" << std::endl << std::endl;
        for (auto line : output)
        {
            std::cerr << line << std::endl;
        }
        std::cerr << std::endl << "expected" << std::endl << std::endl;
        for (auto line : expectedOutput)
        {
            std::cerr << line << std::endl;
        }
        return false;
    }
    bool bRet = true;
    for (unsigned long i = 0; i < output.size(); ++i)
    {
        if (output[i] != expectedOutput[i])
        {
            bRet = false;
            std::cerr << i << ": actual [" << output[i] << "] expected [" << expectedOutput[i] << "]" << std::endl;
        }
    }
    return bRet;
}
