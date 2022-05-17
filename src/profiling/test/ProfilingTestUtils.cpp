//
// Copyright © 2019 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <ArmNNProfilingServiceInitialiser.hpp>
#include "ProfilingOptionsConverter.hpp"
#include "ProfilingTestUtils.hpp"

#include <armnn/Descriptors.hpp>
#include <armnn/profiling/ArmNNProfiling.hpp>

#include <client/src/ProfilingService.hpp>
#include <client/src/ProfilingUtils.hpp>

#include <common/include/Assert.hpp>
#include <common/include/LabelsAndEventClasses.hpp>
#include <common/include/NumericCast.hpp>
#include <common/include/Processes.hpp>
#include <common/include/Threads.hpp>

#include <TestUtils.hpp>

#include <doctest/doctest.h>

uint32_t GetStreamMetaDataPacketSize()
{
    uint32_t sizeUint32 = sizeof(uint32_t);
    uint32_t payloadSize = 0;
    payloadSize += arm::pipe::numeric_cast<uint32_t>(arm::pipe::ARMNN_SOFTWARE_INFO.size()) + 1;
    payloadSize += arm::pipe::numeric_cast<uint32_t>(arm::pipe::ARMNN_HARDWARE_VERSION.size()) + 1;
    payloadSize += arm::pipe::numeric_cast<uint32_t>(arm::pipe::ARMNN_SOFTWARE_VERSION.size()) + 1;
    payloadSize += arm::pipe::numeric_cast<uint32_t>(GetProcessName().size()) + 1;

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
    ARM_PIPE_ASSERT(readableData);

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

ProfilingGuid VerifyTimelineLabelBinaryPacketData(arm::pipe::Optional<ProfilingGuid> guid,
                                                  const std::string& label,
                                                  const unsigned char* readableData,
                                                  unsigned int& offset)
{
    ARM_PIPE_ASSERT(readableData);

    // Utils
    unsigned int uint32_t_size = sizeof(uint32_t);
    unsigned int uint64_t_size = sizeof(uint64_t);
    unsigned int label_size    = arm::pipe::numeric_cast<unsigned int>(label.size());

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
        ArmNNProfilingServiceInitialiser initialiser;
        ProfilingService profilingService(arm::pipe::MAX_ARMNN_COUNTER,
                                          initialiser,
                                          arm::pipe::ARMNN_SOFTWARE_INFO,
                                          arm::pipe::ARMNN_SOFTWARE_VERSION,
                                          arm::pipe::ARMNN_HARDWARE_VERSION);
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
    ARM_PIPE_ASSERT(readableData);

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
                                                arm::pipe::Optional<ProfilingGuid> relationshipGuid,
                                                arm::pipe::Optional<ProfilingGuid> headGuid,
                                                arm::pipe::Optional<ProfilingGuid> tailGuid,
                                                arm::pipe::Optional<ProfilingGuid> attributeGuid,
                                                const unsigned char* readableData,
                                                unsigned int& offset)
{
    ARM_PIPE_ASSERT(readableData);

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

ProfilingGuid VerifyTimelineEntityBinaryPacketData(arm::pipe::Optional<ProfilingGuid> guid,
                                                   const unsigned char* readableData,
                                                   unsigned int& offset)
{
    ARM_PIPE_ASSERT(readableData);

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

ProfilingGuid VerifyTimelineEventBinaryPacket(arm::pipe::Optional<uint64_t> timestamp,
                                              arm::pipe::Optional<int> threadId,
                                              arm::pipe::Optional<ProfilingGuid> eventGuid,
                                              const unsigned char* readableData,
                                              unsigned int& offset)
{
    ARM_PIPE_ASSERT(readableData);

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
        CHECK(readThreadId == arm::pipe::GetCurrentThreadId());
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
    GetProfilingService(&runtime).ResetExternalProfilingOptions(
        ConvertExternalProfilingOptions(options.m_ProfilingOptions), false);

    ArmNNProfilingServiceInitialiser initialiser;
    ProfilingServiceRuntimeHelper profilingServiceHelper(
        arm::pipe::MAX_ARMNN_COUNTER, initialiser, GetProfilingService(&runtime));
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

    armnn::Optional<ConstTensor> optionalBiases;
    std::vector<float> biasesData{ 1.0f, 0.0f, 0.0f };
    ConstTensor biases(biasInfo, biasesData);
    optionalBiases = armnn::Optional<ConstTensor>(biases);

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

    IConnectableLayer* conv2d = net->AddConvolution2dLayer(conv2dDesc);

    armnn::IConnectableLayer* weightsLayer = net->AddConstantLayer(weights, "Weights");
    armnn::IConnectableLayer* biasLayer = net->AddConstantLayer(biases, "Bias");

    weightsLayer->GetOutputSlot(0).SetTensorInfo(weightInfo);
    weightsLayer->GetOutputSlot(0).Connect(conv2d->GetInputSlot(1u));

    biasLayer->GetOutputSlot(0).SetTensorInfo(biasInfo);
    biasLayer->GetOutputSlot(0).Connect(conv2d->GetInputSlot(2u));

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

    BufferManager& bufferManager = profilingServiceHelper.GetProfilingBufferManager();
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
                                               arm::pipe::EmptyOptional(),
                                               optNetGuid,
                                               LabelsAndEventClasses::NETWORK_GUID,
                                               LabelsAndEventClasses::TYPE_GUID,
                                               readableData,
                                               offset);

    // Network - START OF LIFE
    ProfilingGuid networkSolEventGuid = VerifyTimelineEventBinaryPacket(arm::pipe::EmptyOptional(),
                                                                        arm::pipe::EmptyOptional(),
                                                                        arm::pipe::EmptyOptional(),
                                                                        readableData,
                                                                        offset);

    // Network - START OF LIFE event relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::ExecutionLink,
                                               arm::pipe::EmptyOptional(),
                                               optNetGuid,
                                               networkSolEventGuid,
                                               LabelsAndEventClasses::ARMNN_PROFILING_SOL_EVENT_CLASS,
                                               readableData,
                                               offset);

    // Process ID Label
    int processID = arm::pipe::GetCurrentProcessId();
    std::stringstream ss;
    ss << processID;
    std::string processIdLabel = ss.str();
    VerifyTimelineLabelBinaryPacketData(
        arm::pipe::EmptyOptional(), processIdLabel, readableData, offset);

    // Entity - Process ID relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::LabelLink,
                                               arm::pipe::EmptyOptional(),
                                               optNetGuid,
                                               arm::pipe::EmptyOptional(),
                                               LabelsAndEventClasses::PROCESS_ID_GUID,
                                               readableData,
                                               offset);

    // Input layer
    // Input layer entity
    VerifyTimelineEntityBinaryPacketData(input->GetGuid(), readableData, offset);
    // Name Entity
    ProfilingGuid inputLabelGuid = VerifyTimelineLabelBinaryPacketData(
        arm::pipe::EmptyOptional(), "input", readableData, offset);

    // Entity - Name relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::LabelLink,
                                               arm::pipe::EmptyOptional(),
                                               input->GetGuid(),
                                               inputLabelGuid,
                                               LabelsAndEventClasses::NAME_GUID,
                                               readableData,
                                               offset);

    // Entity - Type relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::LabelLink,
                                               arm::pipe::EmptyOptional(),
                                               input->GetGuid(),
                                               LabelsAndEventClasses::LAYER_GUID,
                                               LabelsAndEventClasses::TYPE_GUID,
                                               readableData,
                                               offset);

    // Network - Input layer relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::RetentionLink,
                                               arm::pipe::EmptyOptional(),
                                               optNetGuid,
                                               input->GetGuid(),
                                               LabelsAndEventClasses::CHILD_GUID,
                                               readableData,
                                               offset);

    // Weights layer
    //  We will not check the GUID from the packets since we haven't direct access to the layer
    //  The GUID will change depending on the number of tests ran since we do are not explicitly resetting the
    //  ProfilingGuid counter at the beginning of this test


    // Weights layer entity
    VerifyTimelineEntityBinaryPacketData( arm::pipe::EmptyOptional(), readableData, offset);

    // Name entity
    ProfilingGuid weightsNameLabelGuid = VerifyTimelineLabelBinaryPacketData(
        arm::pipe::EmptyOptional(), "Weights", readableData, offset);

    // Entity - Name relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::LabelLink,
                                               arm::pipe::EmptyOptional(),
                                               arm::pipe::EmptyOptional(),
                                               weightsNameLabelGuid,
                                               LabelsAndEventClasses::NAME_GUID,
                                               readableData,
                                               offset);

    // Entity - Type relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::LabelLink,
                                               arm::pipe::EmptyOptional(),
                                               arm::pipe::EmptyOptional(),
                                               LabelsAndEventClasses::LAYER_GUID,
                                               LabelsAndEventClasses::TYPE_GUID,
                                               readableData,
                                               offset);

    // Network - Weights layer relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::RetentionLink,
                                               arm::pipe::EmptyOptional(),
                                               optNetGuid,
                                               arm::pipe::EmptyOptional(),
                                               LabelsAndEventClasses::CHILD_GUID,
                                               readableData,
                                               offset);

    // Weights workload
    // Weights workload entity
    ProfilingGuid weightsWorkloadGuid = VerifyTimelineEntityBinaryPacketData(
        arm::pipe::EmptyOptional(), readableData, offset);

    // Entity - Type relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::LabelLink,
                                               arm::pipe::EmptyOptional(),
                                               weightsWorkloadGuid,
                                               LabelsAndEventClasses::WORKLOAD_GUID,
                                               LabelsAndEventClasses::TYPE_GUID,
                                               readableData,
                                               offset);

    // BackendId entity
    ProfilingGuid backendIdLabelGuid = VerifyTimelineLabelBinaryPacketData(
        arm::pipe::EmptyOptional(), backendId.Get(), readableData, offset);

    // Entity - BackendId relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::LabelLink,
                                               arm::pipe::EmptyOptional(),
                                               weightsWorkloadGuid,
                                               backendIdLabelGuid,
                                               LabelsAndEventClasses::BACKENDID_GUID,
                                               readableData,
                                               offset);


    // Weights layer - Weights workload relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::RetentionLink,
                                               arm::pipe::EmptyOptional(),
                                               arm::pipe::EmptyOptional(),
                                               weightsWorkloadGuid,
                                               LabelsAndEventClasses::CHILD_GUID,
                                               readableData,
                                               offset);

    // Bias layer
    //  We will not check the GUID from the packets since we haven't direct access to the layer
    //  The GUID will change depending on the number of tests ran since we do are not explicitly resetting the
    //  ProfilingGuid counter at the beginning of this test

    // Bias layer entity
    VerifyTimelineEntityBinaryPacketData(arm::pipe::EmptyOptional(), readableData, offset);

    // Name entity
    ProfilingGuid biasNameLabelGuid = VerifyTimelineLabelBinaryPacketData(
        arm::pipe::EmptyOptional(), "Bias", readableData, offset);

    // Entity - Name relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::LabelLink,
                                               arm::pipe::EmptyOptional(),
                                               arm::pipe::EmptyOptional(),
                                               biasNameLabelGuid,
                                               LabelsAndEventClasses::NAME_GUID,
                                               readableData,
                                               offset);

    // Entity - Type relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::LabelLink,
                                               arm::pipe::EmptyOptional(),
                                               arm::pipe::EmptyOptional(),
                                               LabelsAndEventClasses::LAYER_GUID,
                                               LabelsAndEventClasses::TYPE_GUID,
                                               readableData,
                                               offset);

    // Network - Bias layer relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::RetentionLink,
                                               arm::pipe::EmptyOptional(),
                                               optNetGuid,
                                               arm::pipe::EmptyOptional(),
                                               LabelsAndEventClasses::CHILD_GUID,
                                               readableData,
                                               offset);

    // Bias workload
    // Bias workload entity
    ProfilingGuid biasWorkloadGuid = VerifyTimelineEntityBinaryPacketData(
        arm::pipe::EmptyOptional(), readableData, offset);

    // Entity - Type relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::LabelLink,
                                               arm::pipe::EmptyOptional(),
                                               biasWorkloadGuid,
                                               LabelsAndEventClasses::WORKLOAD_GUID,
                                               LabelsAndEventClasses::TYPE_GUID,
                                               readableData,
                                               offset);

    // BackendId entity
    backendIdLabelGuid = VerifyTimelineLabelBinaryPacketData(
        arm::pipe::EmptyOptional(), backendId.Get(), readableData, offset);

    // Entity - BackendId relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::LabelLink,
                                               arm::pipe::EmptyOptional(),
                                               biasWorkloadGuid,
                                               backendIdLabelGuid,
                                               LabelsAndEventClasses::BACKENDID_GUID,
                                               readableData,
                                               offset);


    // Bias layer - Bias workload relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::RetentionLink,
                                               arm::pipe::EmptyOptional(),
                                               arm::pipe::EmptyOptional(),
                                               biasWorkloadGuid,
                                               LabelsAndEventClasses::CHILD_GUID,
                                               readableData,
                                               offset);

    // Conv2d layer
    // Conv2d layer entity
    VerifyTimelineEntityBinaryPacketData(conv2d->GetGuid(), readableData, offset);

    // Name entity
    ProfilingGuid conv2dNameLabelGuid = VerifyTimelineLabelBinaryPacketData(
        arm::pipe::EmptyOptional(), "<Unnamed>", readableData, offset);

    // Entity - Name relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::LabelLink,
                                               arm::pipe::EmptyOptional(),
                                               conv2d->GetGuid(),
                                               conv2dNameLabelGuid,
                                               LabelsAndEventClasses::NAME_GUID,
                                               readableData,
                                               offset);

    // Entity - Type relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::LabelLink,
                                               arm::pipe::EmptyOptional(),
                                               conv2d->GetGuid(),
                                               LabelsAndEventClasses::LAYER_GUID,
                                               LabelsAndEventClasses::TYPE_GUID,
                                               readableData,
                                               offset);

    // Network - Conv2d layer relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::RetentionLink,
                                               arm::pipe::EmptyOptional(),
                                               optNetGuid,
                                               conv2d->GetGuid(),
                                               LabelsAndEventClasses::CHILD_GUID,
                                               readableData,
                                               offset);

    // Input layer - Conv2d layer relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::RetentionLink,
                                               arm::pipe::EmptyOptional(),
                                               input->GetGuid(),
                                               conv2d->GetGuid(),
                                               LabelsAndEventClasses::CONNECTION_GUID,
                                               readableData,
                                               offset);

    // Weights layer - Conv2d layer relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::RetentionLink,
                                               arm::pipe::EmptyOptional(),
                                               arm::pipe::EmptyOptional(),
                                               conv2d->GetGuid(),
                                               LabelsAndEventClasses::CONNECTION_GUID,
                                               readableData,
                                               offset);

    // Bias layer - Conv2d layer relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::RetentionLink,
                                               arm::pipe::EmptyOptional(),
                                               arm::pipe::EmptyOptional(),
                                               conv2d->GetGuid(),
                                               LabelsAndEventClasses::CONNECTION_GUID,
                                               readableData,
                                               offset);

    // Conv2d workload
    // Conv2d workload entity
    ProfilingGuid conv2DWorkloadGuid = VerifyTimelineEntityBinaryPacketData(
        arm::pipe::EmptyOptional(), readableData, offset);

    // Entity - Type relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::LabelLink,
                                               arm::pipe::EmptyOptional(),
                                               conv2DWorkloadGuid,
                                               LabelsAndEventClasses::WORKLOAD_GUID,
                                               LabelsAndEventClasses::TYPE_GUID,
                                               readableData,
                                               offset);

    // BackendId entity
    backendIdLabelGuid = VerifyTimelineLabelBinaryPacketData(
        arm::pipe::EmptyOptional(), backendId.Get(), readableData, offset);

    // Entity - BackendId relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::LabelLink,
                                               arm::pipe::EmptyOptional(),
                                               conv2DWorkloadGuid,
                                               backendIdLabelGuid,
                                               LabelsAndEventClasses::BACKENDID_GUID,
                                               readableData,
                                               offset);


    // Conv2d layer - Conv2d workload relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::RetentionLink,
                                               arm::pipe::EmptyOptional(),
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
        arm::pipe::EmptyOptional(), "abs", readableData, offset);

    // Entity - Name relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::LabelLink,
                                               arm::pipe::EmptyOptional(),
                                               abs->GetGuid(),
                                               absLabelGuid,
                                               LabelsAndEventClasses::NAME_GUID,
                                               readableData,
                                               offset);

    // Entity - Type relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::LabelLink,
                                               arm::pipe::EmptyOptional(),
                                               abs->GetGuid(),
                                               LabelsAndEventClasses::LAYER_GUID,
                                               LabelsAndEventClasses::TYPE_GUID,
                                               readableData,
                                               offset);

    // Network - Abs layer relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::RetentionLink,
                                               arm::pipe::EmptyOptional(),
                                               optNetGuid,
                                               abs->GetGuid(),
                                               LabelsAndEventClasses::CHILD_GUID,
                                               readableData,
                                               offset);

    // Conv2d layer - Abs layer relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::RetentionLink,
                                               arm::pipe::EmptyOptional(),
                                               conv2d->GetGuid(),
                                               abs->GetGuid(),
                                               LabelsAndEventClasses::CONNECTION_GUID,
                                               readableData,
                                               offset);

    // Abs workload
    // Abs workload entity
    ProfilingGuid absWorkloadGuid = VerifyTimelineEntityBinaryPacketData(
        arm::pipe::EmptyOptional(), readableData, offset);

    // Entity - Type relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::LabelLink,
                                               arm::pipe::EmptyOptional(),
                                               absWorkloadGuid,
                                               LabelsAndEventClasses::WORKLOAD_GUID,
                                               LabelsAndEventClasses::TYPE_GUID,
                                               readableData,
                                               offset);

    // BackendId entity
    VerifyTimelineLabelBinaryPacketData(arm::pipe::EmptyOptional(), backendId.Get(), readableData, offset);

    // Entity - BackendId relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::LabelLink,
                                               arm::pipe::EmptyOptional(),
                                               absWorkloadGuid,
                                               backendIdLabelGuid,
                                               LabelsAndEventClasses::BACKENDID_GUID,
                                               readableData,
                                               offset);

    // Abs layer - Abs workload relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::RetentionLink,
                                               arm::pipe::EmptyOptional(),
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
        arm::pipe::EmptyOptional(), "output", readableData, offset);

    // Entity - Name relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::LabelLink,
                                               arm::pipe::EmptyOptional(),
                                               output->GetGuid(),
                                               outputLabelGuid,
                                               LabelsAndEventClasses::NAME_GUID,
                                               readableData,
                                               offset);

    // Entity - Type relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::LabelLink,
                                               arm::pipe::EmptyOptional(),
                                               output->GetGuid(),
                                               LabelsAndEventClasses::LAYER_GUID,
                                               LabelsAndEventClasses::TYPE_GUID,
                                               readableData,
                                               offset);

    // Network - Output layer relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::RetentionLink,
                                               arm::pipe::EmptyOptional(),
                                               optNetGuid,
                                               output->GetGuid(),
                                               LabelsAndEventClasses::CHILD_GUID,
                                               readableData,
                                               offset);

    // Abs layer - Output layer relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::RetentionLink,
                                               arm::pipe::EmptyOptional(),
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
    ProfilingGuid inputWorkloadGuid = VerifyTimelineEntityBinaryPacketData(
        arm::pipe::EmptyOptional(), readableData, offset);

    // Entity - Type relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::LabelLink,
                                               arm::pipe::EmptyOptional(),
                                               inputWorkloadGuid,
                                               LabelsAndEventClasses::WORKLOAD_GUID,
                                               LabelsAndEventClasses::TYPE_GUID,
                                               readableData,
                                               offset);

    // BackendId entity
    VerifyTimelineLabelBinaryPacketData(arm::pipe::EmptyOptional(), backendId.Get(), readableData, offset);

    // Entity - BackendId relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::LabelLink,
                                               arm::pipe::EmptyOptional(),
                                               inputWorkloadGuid,
                                               backendIdLabelGuid,
                                               LabelsAndEventClasses::BACKENDID_GUID,
                                               readableData,
                                               offset);

    // Input layer - Input workload relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::RetentionLink,
                                               arm::pipe::EmptyOptional(),
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
    ProfilingGuid outputWorkloadGuid = VerifyTimelineEntityBinaryPacketData(
        arm::pipe::EmptyOptional(), readableData, offset);

    // Entity - Type relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::LabelLink,
                                               arm::pipe::EmptyOptional(),
                                               outputWorkloadGuid,
                                               LabelsAndEventClasses::WORKLOAD_GUID,
                                               LabelsAndEventClasses::TYPE_GUID,
                                               readableData,
                                               offset);

    // BackendId entity
    VerifyTimelineLabelBinaryPacketData(arm::pipe::EmptyOptional(), backendId.Get(), readableData, offset);

    // Entity - BackendId relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::LabelLink,
                                               arm::pipe::EmptyOptional(),
                                               outputWorkloadGuid,
                                               backendIdLabelGuid,
                                               LabelsAndEventClasses::BACKENDID_GUID,
                                               readableData,
                                               offset);

    // Output layer - Output workload relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::RetentionLink,
                                               arm::pipe::EmptyOptional(),
                                               output->GetGuid(),
                                               outputWorkloadGuid,
                                               LabelsAndEventClasses::CHILD_GUID,
                                               readableData,
                                               offset);

    bufferManager.MarkRead(outputReadableBuffer);

    // Validate inference data
    size = inferenceReadableBuffer->GetSize();

    CHECK(size == 1748 + 10 * ThreadIdSize);

    readableData = inferenceReadableBuffer->GetReadableData();
    CHECK(readableData != nullptr);

    offset = 0;

    // Verify Header
    VerifyTimelineHeaderBinary(readableData, offset, 1740 + 10 * ThreadIdSize);

    // Inference timeline trace
    // Inference entity
    ProfilingGuid inferenceGuid = VerifyTimelineEntityBinaryPacketData(
        arm::pipe::EmptyOptional(), readableData, offset);

    // Entity - Type relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::LabelLink,
                                               arm::pipe::EmptyOptional(),
                                               inferenceGuid,
                                               LabelsAndEventClasses::INFERENCE_GUID,
                                               LabelsAndEventClasses::TYPE_GUID,
                                               readableData,
                                               offset);

    // Network - Inference relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::RetentionLink,
                                               arm::pipe::EmptyOptional(),
                                               optNetGuid,
                                               inferenceGuid,
                                               LabelsAndEventClasses::EXECUTION_OF_GUID,
                                               readableData,
                                               offset);

    // Start Inference life
    // Event packet - timeline, threadId, eventGuid
    ProfilingGuid inferenceEventGuid = VerifyTimelineEventBinaryPacket(
        arm::pipe::EmptyOptional(), arm::pipe::EmptyOptional(), arm::pipe::EmptyOptional(), readableData, offset);

    // Inference - event relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::ExecutionLink,
                                               arm::pipe::EmptyOptional(),
                                               inferenceGuid,
                                               inferenceEventGuid,
                                               LabelsAndEventClasses::ARMNN_PROFILING_SOL_EVENT_CLASS,
                                               readableData,
                                               offset);

    // Execution
    // Input workload execution
    // Input workload execution entity
    ProfilingGuid inputWorkloadExecutionGuid = VerifyTimelineEntityBinaryPacketData(
        arm::pipe::EmptyOptional(), readableData, offset);

    // Entity - Type relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::LabelLink,
                                               arm::pipe::EmptyOptional(),
                                               inputWorkloadExecutionGuid,
                                               LabelsAndEventClasses::WORKLOAD_EXECUTION_GUID,
                                               LabelsAndEventClasses::TYPE_GUID,
                                               readableData,
                                               offset);

    // Inference - Workload execution relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::RetentionLink,
                                               arm::pipe::EmptyOptional(),
                                               inferenceGuid,
                                               inputWorkloadExecutionGuid,
                                               LabelsAndEventClasses::CHILD_GUID,
                                               readableData,
                                               offset);

    // Workload - Workload execution relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::RetentionLink,
                                               arm::pipe::EmptyOptional(),
                                               inputWorkloadGuid,
                                               inputWorkloadExecutionGuid,
                                               LabelsAndEventClasses::EXECUTION_OF_GUID,
                                               readableData,
                                               offset);

    // Start Input workload execution life
    // Event packet - timeline, threadId, eventGuid
    ProfilingGuid inputWorkloadExecutionSOLEventId = VerifyTimelineEventBinaryPacket(
        arm::pipe::EmptyOptional(), arm::pipe::EmptyOptional(), arm::pipe::EmptyOptional(), readableData, offset);

    // Input workload execution - event relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::ExecutionLink,
                                               arm::pipe::EmptyOptional(),
                                               inputWorkloadExecutionGuid,
                                               inputWorkloadExecutionSOLEventId,
                                               LabelsAndEventClasses::ARMNN_PROFILING_SOL_EVENT_CLASS,
                                               readableData,
                                               offset);

    // End of Input workload execution life
    // Event packet - timeline, threadId, eventGuid
    ProfilingGuid inputWorkloadExecutionEOLEventId = VerifyTimelineEventBinaryPacket(
        arm::pipe::EmptyOptional(), arm::pipe::EmptyOptional(), arm::pipe::EmptyOptional(), readableData, offset);

    // Input workload execution - event relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::ExecutionLink,
                                               arm::pipe::EmptyOptional(),
                                               inputWorkloadExecutionGuid,
                                               inputWorkloadExecutionEOLEventId,
                                               LabelsAndEventClasses::ARMNN_PROFILING_EOL_EVENT_CLASS,
                                               readableData,
                                               offset);

   // Weights workload execution
    // Weights workload execution entity
    ProfilingGuid weightsWorkloadExecutionGuid = VerifyTimelineEntityBinaryPacketData(
        arm::pipe::EmptyOptional(), readableData, offset);

    // Entity - Type relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::LabelLink,
                                               arm::pipe::EmptyOptional(),
                                               weightsWorkloadExecutionGuid,
                                               LabelsAndEventClasses::WORKLOAD_EXECUTION_GUID,
                                               LabelsAndEventClasses::TYPE_GUID,
                                               readableData,
                                               offset);

    // Inference - Workload execution relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::RetentionLink,
                                               arm::pipe::EmptyOptional(),
                                               inferenceGuid,
                                               weightsWorkloadExecutionGuid,
                                               LabelsAndEventClasses::CHILD_GUID,
                                               readableData,
                                               offset);

    // Workload - Workload execution relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::RetentionLink,
                                               arm::pipe::EmptyOptional(),
                                               weightsWorkloadGuid,
                                               weightsWorkloadExecutionGuid,
                                               LabelsAndEventClasses::EXECUTION_OF_GUID,
                                               readableData,
                                               offset);

    // Start Weights workload execution life
    // Event packet - timeline, threadId, eventGuid
    ProfilingGuid weightsWorkloadExecutionSOLEventGuid = VerifyTimelineEventBinaryPacket(
        arm::pipe::EmptyOptional(), arm::pipe::EmptyOptional(), arm::pipe::EmptyOptional(), readableData, offset);

    // Weights workload execution - event relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::ExecutionLink,
                                               arm::pipe::EmptyOptional(),
                                               weightsWorkloadExecutionGuid,
                                               weightsWorkloadExecutionSOLEventGuid,
                                               LabelsAndEventClasses::ARMNN_PROFILING_SOL_EVENT_CLASS,
                                               readableData,
                                               offset);

    // End of Weights workload execution life
    // Event packet - timeline, threadId, eventGuid
    ProfilingGuid weightsWorkloadExecutionEOLEventGuid = VerifyTimelineEventBinaryPacket(
        arm::pipe::EmptyOptional(), arm::pipe::EmptyOptional(), arm::pipe::EmptyOptional(), readableData, offset);

    // Weights workload execution - event relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::ExecutionLink,
                                               arm::pipe::EmptyOptional(),
                                               weightsWorkloadExecutionGuid,
                                               weightsWorkloadExecutionEOLEventGuid,
                                               LabelsAndEventClasses::ARMNN_PROFILING_EOL_EVENT_CLASS,
                                               readableData,
                                               offset);

   // Bias workload execution
    // Bias workload execution entity
    ProfilingGuid biasWorkloadExecutionGuid = VerifyTimelineEntityBinaryPacketData(
        arm::pipe::EmptyOptional(), readableData, offset);

    // Entity - Type relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::LabelLink,
                                               arm::pipe::EmptyOptional(),
                                               biasWorkloadExecutionGuid,
                                               LabelsAndEventClasses::WORKLOAD_EXECUTION_GUID,
                                               LabelsAndEventClasses::TYPE_GUID,
                                               readableData,
                                               offset);

    // Inference - Workload execution relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::RetentionLink,
                                               arm::pipe::EmptyOptional(),
                                               inferenceGuid,
                                               biasWorkloadExecutionGuid,
                                               LabelsAndEventClasses::CHILD_GUID,
                                               readableData,
                                               offset);

    // Workload - Workload execution relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::RetentionLink,
                                               arm::pipe::EmptyOptional(),
                                               biasWorkloadGuid,
                                               biasWorkloadExecutionGuid,
                                               LabelsAndEventClasses::EXECUTION_OF_GUID,
                                               readableData,
                                               offset);

    // Start Bias workload execution life
    // Event packet - timeline, threadId, eventGuid
    ProfilingGuid biasWorkloadExecutionSOLEventGuid = VerifyTimelineEventBinaryPacket(
        arm::pipe::EmptyOptional(), arm::pipe::EmptyOptional(), arm::pipe::EmptyOptional(), readableData, offset);

    // Bias workload execution - event relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::ExecutionLink,
                                               arm::pipe::EmptyOptional(),
                                               biasWorkloadExecutionGuid,
                                               biasWorkloadExecutionSOLEventGuid,
                                               LabelsAndEventClasses::ARMNN_PROFILING_SOL_EVENT_CLASS,
                                               readableData,
                                               offset);

    // End of Bias workload execution life
    // Event packet - timeline, threadId, eventGuid
    ProfilingGuid biasWorkloadExecutionEOLEventGuid = VerifyTimelineEventBinaryPacket(
        arm::pipe::EmptyOptional(), arm::pipe::EmptyOptional(), arm::pipe::EmptyOptional(), readableData, offset);

    // Bias workload execution - event relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::ExecutionLink,
                                               arm::pipe::EmptyOptional(),
                                               biasWorkloadExecutionGuid,
                                               biasWorkloadExecutionEOLEventGuid,
                                               LabelsAndEventClasses::ARMNN_PROFILING_EOL_EVENT_CLASS,
                                               readableData,
                                               offset);

    // Conv2d workload execution
    // Conv2d workload execution entity
    ProfilingGuid conv2DWorkloadExecutionGuid = VerifyTimelineEntityBinaryPacketData(
        arm::pipe::EmptyOptional(), readableData, offset);

    // Entity - Type relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::LabelLink,
                                               arm::pipe::EmptyOptional(),
                                               conv2DWorkloadExecutionGuid,
                                               LabelsAndEventClasses::WORKLOAD_EXECUTION_GUID,
                                               LabelsAndEventClasses::TYPE_GUID,
                                               readableData,
                                               offset);

    // Inference - Workload execution relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::RetentionLink,
                                               arm::pipe::EmptyOptional(),
                                               inferenceGuid,
                                               conv2DWorkloadExecutionGuid,
                                               LabelsAndEventClasses::CHILD_GUID,
                                               readableData,
                                               offset);

    // Workload - Workload execution relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::RetentionLink,
                                               arm::pipe::EmptyOptional(),
                                               conv2DWorkloadGuid,
                                               conv2DWorkloadExecutionGuid,
                                               LabelsAndEventClasses::EXECUTION_OF_GUID,
                                               readableData,
                                               offset);

    // Start Conv2d workload execution life
    // Event packet - timeline, threadId, eventGuid
    ProfilingGuid conv2DWorkloadExecutionSOLEventGuid = VerifyTimelineEventBinaryPacket(
        arm::pipe::EmptyOptional(), arm::pipe::EmptyOptional(), arm::pipe::EmptyOptional(), readableData, offset);

    // Conv2d workload execution - event relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::ExecutionLink,
                                               arm::pipe::EmptyOptional(),
                                               conv2DWorkloadExecutionGuid,
                                               conv2DWorkloadExecutionSOLEventGuid,
                                               LabelsAndEventClasses::ARMNN_PROFILING_SOL_EVENT_CLASS,
                                               readableData,
                                               offset);

    // End of Conv2d workload execution life
    // Event packet - timeline, threadId, eventGuid
    ProfilingGuid conv2DWorkloadExecutionEOLEventGuid = VerifyTimelineEventBinaryPacket(
        arm::pipe::EmptyOptional(), arm::pipe::EmptyOptional(), arm::pipe::EmptyOptional(), readableData, offset);

    // Conv2d workload execution - event relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::ExecutionLink,
                                               arm::pipe::EmptyOptional(),
                                               conv2DWorkloadExecutionGuid,
                                               conv2DWorkloadExecutionEOLEventGuid,
                                               LabelsAndEventClasses::ARMNN_PROFILING_EOL_EVENT_CLASS,
                                               readableData,
                                               offset);

    // Abs workload execution
    // Abs workload execution entity
    ProfilingGuid absWorkloadExecutionGuid = VerifyTimelineEntityBinaryPacketData(
        arm::pipe::EmptyOptional(), readableData, offset);

    // Entity - Type relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::LabelLink,
                                               arm::pipe::EmptyOptional(),
                                               absWorkloadExecutionGuid,
                                               LabelsAndEventClasses::WORKLOAD_EXECUTION_GUID,
                                               LabelsAndEventClasses::TYPE_GUID,
                                               readableData,
                                               offset);

    // Inference - Workload execution relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::RetentionLink,
                                               arm::pipe::EmptyOptional(),
                                               inferenceGuid,
                                               absWorkloadExecutionGuid,
                                               LabelsAndEventClasses::CHILD_GUID,
                                               readableData,
                                               offset);

    // Workload - Workload execution relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::RetentionLink,
                                               arm::pipe::EmptyOptional(),
                                               absWorkloadGuid,
                                               absWorkloadExecutionGuid,
                                               LabelsAndEventClasses::EXECUTION_OF_GUID,
                                               readableData,
                                               offset);

    // Start Abs workload execution life
    // Event packet - timeline, threadId, eventGuid
    ProfilingGuid absWorkloadExecutionSOLEventGuid = VerifyTimelineEventBinaryPacket(
        arm::pipe::EmptyOptional(), arm::pipe::EmptyOptional(), arm::pipe::EmptyOptional(), readableData, offset);

    // Abs workload execution - event relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::ExecutionLink,
                                               arm::pipe::EmptyOptional(),
                                               absWorkloadExecutionGuid,
                                               absWorkloadExecutionSOLEventGuid,
                                               LabelsAndEventClasses::ARMNN_PROFILING_SOL_EVENT_CLASS,
                                               readableData,
                                               offset);

    // End of Abs workload execution life
    // Event packet - timeline, threadId, eventGuid
    ProfilingGuid absWorkloadExecutionEOLEventGuid = VerifyTimelineEventBinaryPacket(
        arm::pipe::EmptyOptional(), arm::pipe::EmptyOptional(), arm::pipe::EmptyOptional(), readableData, offset);

    // Abs workload execution - event relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::ExecutionLink,
                                               arm::pipe::EmptyOptional(),
                                               absWorkloadExecutionGuid,
                                               absWorkloadExecutionEOLEventGuid,
                                               LabelsAndEventClasses::ARMNN_PROFILING_EOL_EVENT_CLASS,
                                               readableData,
                                               offset);

    // Output workload execution
    // Output workload execution entity
    ProfilingGuid outputWorkloadExecutionGuid = VerifyTimelineEntityBinaryPacketData(
        arm::pipe::EmptyOptional(), readableData, offset);

    // Entity - Type relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::LabelLink,
                                               arm::pipe::EmptyOptional(),
                                               outputWorkloadExecutionGuid,
                                               LabelsAndEventClasses::WORKLOAD_EXECUTION_GUID,
                                               LabelsAndEventClasses::TYPE_GUID,
                                               readableData,
                                               offset);

    // Inference - Workload execution relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::RetentionLink,
                                               arm::pipe::EmptyOptional(),
                                               inferenceGuid,
                                               outputWorkloadExecutionGuid,
                                               LabelsAndEventClasses::CHILD_GUID,
                                               readableData,
                                               offset);

    // Workload - Workload execution relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::RetentionLink,
                                               arm::pipe::EmptyOptional(),
                                               outputWorkloadGuid,
                                               outputWorkloadExecutionGuid,
                                               LabelsAndEventClasses::EXECUTION_OF_GUID,
                                               readableData,
                                               offset);

    // Start Output workload execution life
    // Event packet - timeline, threadId, eventGuid
    ProfilingGuid outputWorkloadExecutionSOLEventGuid = VerifyTimelineEventBinaryPacket(
        arm::pipe::EmptyOptional(), arm::pipe::EmptyOptional(), arm::pipe::EmptyOptional(), readableData, offset);

    // Output workload execution - event relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::ExecutionLink,
                                               arm::pipe::EmptyOptional(),
                                               outputWorkloadExecutionGuid,
                                               outputWorkloadExecutionSOLEventGuid,
                                               LabelsAndEventClasses::ARMNN_PROFILING_SOL_EVENT_CLASS,
                                               readableData,
                                               offset);

    // End of Normalize workload execution life
    // Event packet - timeline, threadId, eventGuid
    ProfilingGuid outputWorkloadExecutionEOLEventGuid = VerifyTimelineEventBinaryPacket(
        arm::pipe::EmptyOptional(), arm::pipe::EmptyOptional(), arm::pipe::EmptyOptional(), readableData, offset);

    // Output workload execution - event relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::ExecutionLink,
                                               arm::pipe::EmptyOptional(),
                                               outputWorkloadExecutionGuid,
                                               outputWorkloadExecutionEOLEventGuid,
                                               LabelsAndEventClasses::ARMNN_PROFILING_EOL_EVENT_CLASS,
                                               readableData,
                                               offset);

    // End of Inference life
    // Event packet - timeline, threadId, eventGuid
    ProfilingGuid inferenceEOLEventGuid = VerifyTimelineEventBinaryPacket(
        arm::pipe::EmptyOptional(), arm::pipe::EmptyOptional(), arm::pipe::EmptyOptional(), readableData, offset);

    // Inference - event relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::ExecutionLink,
                                               arm::pipe::EmptyOptional(),
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
