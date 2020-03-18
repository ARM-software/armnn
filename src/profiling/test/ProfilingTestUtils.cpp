//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ProfilingTestUtils.hpp"
#include "ProfilingUtils.hpp"

#include <armnn/Descriptors.hpp>
#include <LabelsAndEventClasses.hpp>
#include <ProfilingService.hpp>
#include <Runtime.hpp>

#include <boost/test/unit_test.hpp>

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
    BOOST_ASSERT(readableData);

    // Utils
    unsigned int uint32_t_size = sizeof(uint32_t);

    // Check the TimelineEventClassBinaryPacket header
    uint32_t entityBinaryPacketHeaderWord0 = ReadUint32(readableData, offset);
    uint32_t entityBinaryPacketFamily      = (entityBinaryPacketHeaderWord0 >> 26) & 0x0000003F;
    uint32_t entityBinaryPacketClass       = (entityBinaryPacketHeaderWord0 >> 19) & 0x0000007F;
    uint32_t entityBinaryPacketType        = (entityBinaryPacketHeaderWord0 >> 16) & 0x00000007;
    uint32_t entityBinaryPacketStreamId    = (entityBinaryPacketHeaderWord0 >>  0) & 0x00000007;
    BOOST_CHECK(entityBinaryPacketFamily   == 1);
    BOOST_CHECK(entityBinaryPacketClass    == 0);
    BOOST_CHECK(entityBinaryPacketType     == 1);
    BOOST_CHECK(entityBinaryPacketStreamId == 0);
    offset += uint32_t_size;
    uint32_t entityBinaryPacketHeaderWord1   = ReadUint32(readableData, offset);
    uint32_t eventBinaryPacketSequenceNumber = (entityBinaryPacketHeaderWord1 >> 24) & 0x00000001;
    uint32_t eventBinaryPacketDataLength     = (entityBinaryPacketHeaderWord1 >>  0) & 0x00FFFFFF;
    BOOST_CHECK(eventBinaryPacketSequenceNumber == 0);
    BOOST_CHECK(eventBinaryPacketDataLength     == packetDataLength);
    offset += uint32_t_size;
}

void VerifyTimelineLabelBinaryPacketData(Optional<ProfilingGuid> guid,
                                         const std::string& label,
                                         const unsigned char* readableData,
                                         unsigned int& offset)
{
    BOOST_ASSERT(readableData);

    // Utils
    unsigned int uint32_t_size = sizeof(uint32_t);
    unsigned int uint64_t_size = sizeof(uint64_t);
    unsigned int label_size    = boost::numeric_cast<unsigned int>(label.size());

    // Check the decl id
    uint32_t eventClassDeclId = ReadUint32(readableData, offset);
    BOOST_CHECK(eventClassDeclId == 0);

    // Check the profiling GUID
    offset += uint32_t_size;
    uint64_t readProfilingGuid = ReadUint64(readableData, offset);
    if (guid.has_value())
    {
        BOOST_CHECK(readProfilingGuid == guid.value());
    }
    else
    {
        armnn::profiling::ProfilingService profilingService;
        BOOST_CHECK(readProfilingGuid == profilingService.GetStaticId(label));
    }

    // Check the SWTrace label
    offset += uint64_t_size;
    uint32_t swTraceLabelLength = ReadUint32(readableData, offset);
    BOOST_CHECK(swTraceLabelLength == label_size + 1);               // Label length including the null-terminator
    offset += uint32_t_size;
    BOOST_CHECK(std::memcmp(readableData + offset,                  // Offset to the label in the buffer
                               label.data(),                           // The original label
                               swTraceLabelLength - 1) == 0);          // The length of the label

    // SWTrace strings are written in blocks of words, so the offset has to be updated to the next whole word
    offset += OffsetToNextWord(swTraceLabelLength);
}

void VerifyTimelineEventClassBinaryPacketData(ProfilingGuid guid,
                                              const unsigned char* readableData,
                                              unsigned int& offset)
{
    BOOST_ASSERT(readableData);

    // Utils
    unsigned int uint32_t_size = sizeof(uint32_t);
    unsigned int uint64_t_size = sizeof(uint64_t);

    // Check the decl id
    uint32_t eventClassDeclId = ReadUint32(readableData, offset);
    BOOST_CHECK(eventClassDeclId == 2);

    // Check the profiling GUID
    offset += uint32_t_size;
    uint64_t readProfilingGuid = ReadUint64(readableData, offset);
    BOOST_CHECK(readProfilingGuid == guid);

    // Update the offset to allow parsing to be continued after this function returns
    offset += uint64_t_size;
}

void VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType relationshipType,
                                            Optional<ProfilingGuid> relationshipGuid,
                                            Optional<ProfilingGuid> headGuid,
                                            Optional<ProfilingGuid> tailGuid,
                                            const unsigned char* readableData,
                                            unsigned int& offset)
{
    BOOST_ASSERT(readableData);

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
            BOOST_ERROR("Unknown relationship type");
    }

    // Utils
    unsigned int uint32_t_size = sizeof(uint32_t);
    unsigned int uint64_t_size = sizeof(uint64_t);

    // Check the decl id
    uint32_t eventClassDeclId = ReadUint32(readableData, offset);
    BOOST_CHECK(eventClassDeclId == 3);

    // Check the relationship type
    offset += uint32_t_size;
    uint32_t readRelationshipTypeUint = ReadUint32(readableData, offset);
    BOOST_CHECK(readRelationshipTypeUint == relationshipTypeUint);

    // Check the relationship GUID
    offset += uint32_t_size;
    uint64_t readRelationshipGuid = ReadUint64(readableData, offset);
    if (relationshipGuid.has_value())
    {
        BOOST_CHECK(readRelationshipGuid == relationshipGuid.value());
    }
    else
    {
        BOOST_CHECK(readRelationshipGuid != ProfilingGuid(0));
    }

    // Check the head of relationship GUID
    offset += uint64_t_size;
    uint64_t readHeadRelationshipGuid = ReadUint64(readableData, offset);
    if (headGuid.has_value())
    {
        BOOST_CHECK(readHeadRelationshipGuid == headGuid.value());
    }
    else
    {
        BOOST_CHECK(readHeadRelationshipGuid != ProfilingGuid(0));
    }

    // Check the tail of relationship GUID
    offset += uint64_t_size;
    uint64_t readTailRelationshipGuid = ReadUint64(readableData, offset);
    if (tailGuid.has_value())
    {
        BOOST_CHECK(readTailRelationshipGuid == tailGuid.value());
    }
    else
    {
        BOOST_CHECK(readTailRelationshipGuid != ProfilingGuid(0));
    }

    // Update the offset to allow parsing to be continued after this function returns
    offset += uint64_t_size;
}

void VerifyTimelineEntityBinaryPacketData(Optional<ProfilingGuid> guid,
                                          const unsigned char* readableData,
                                          unsigned int& offset)
{
    BOOST_ASSERT(readableData);

    // Utils
    unsigned int uint32_t_size = sizeof(uint32_t);
    unsigned int uint64_t_size = sizeof(uint64_t);

    // Reading TimelineEntityClassBinaryPacket
    // Check the decl_id
    uint32_t entityDeclId = ReadUint32(readableData, offset);
    BOOST_CHECK(entityDeclId == 1);

    // Check the profiling GUID
    offset += uint32_t_size;
    uint64_t readProfilingGuid = ReadUint64(readableData, offset);

    if (guid.has_value())
    {
        BOOST_CHECK(readProfilingGuid == guid.value());
    }
    else
    {
        BOOST_CHECK(readProfilingGuid != ProfilingGuid(0));
    }

    offset += uint64_t_size;
}

void VerifyTimelineEventBinaryPacket(Optional<uint64_t> timestamp,
                                     Optional<std::thread::id> threadId,
                                     Optional<ProfilingGuid> eventGuid,
                                     const unsigned char* readableData,
                                     unsigned int& offset)
{
    BOOST_ASSERT(readableData);

    // Utils
    unsigned int uint32_t_size = sizeof(uint32_t);
    unsigned int uint64_t_size = sizeof(uint64_t);
    unsigned int threadId_size = sizeof(std::thread::id);

    // Reading TimelineEventBinaryPacket
    // Check the decl_id
    uint32_t entityDeclId = ReadUint32(readableData, offset);
    BOOST_CHECK(entityDeclId == 4);

    // Check the timestamp
    offset += uint32_t_size;
    uint64_t readTimestamp = ReadUint64(readableData, offset);
    if (timestamp.has_value())
    {
        BOOST_CHECK(readTimestamp == timestamp.value());
    }
    else
    {
        BOOST_CHECK(readTimestamp != 0);
    }

    // Check the thread id
    offset += uint64_t_size;
    std::vector<uint8_t> readThreadId(threadId_size, 0);
    ReadBytes(readableData, offset, threadId_size, readThreadId.data());
    if (threadId.has_value())
    {
        BOOST_CHECK(readThreadId == threadId.value());
    }
    else
    {
        BOOST_CHECK(readThreadId == std::this_thread::get_id());
    }

    // Check the event GUID
    offset += threadId_size;
    uint64_t readEventGuid = ReadUint64(readableData, offset);
    if (eventGuid.has_value())
    {
        BOOST_CHECK(readEventGuid == eventGuid.value());
    }
    else
    {
        BOOST_CHECK(readEventGuid != ProfilingGuid(0));
    }

    offset += uint64_t_size;
}

void VerifyPostOptimisationStructureTestImpl(armnn::BackendId backendId)
{
    using namespace armnn;

    // Create runtime in which test will run
    armnn::IRuntime::CreationOptions options;
    options.m_ProfilingOptions.m_EnableProfiling = true;
//    armnn::IRuntimePtr runtime(armnn::IRuntime::Create(options));
    armnn::Runtime runtime(options);

    // build up the structure of the network
    INetworkPtr net(INetwork::Create());

    // Convolution details
    TensorInfo inputInfo({ 1, 2, 5, 1 }, DataType::Float32);
    TensorInfo weightInfo({ 3, 2, 3, 1 }, DataType::Float32);
    TensorInfo biasInfo({ 3 }, DataType::Float32);
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

    // Activation layer
    armnn::ActivationDescriptor activationDesc;
    armnn::IConnectableLayer* const activation = net->AddActivationLayer(activationDesc, "activation");

    // Output layer
    IConnectableLayer* output = net->AddOutputLayer(0, "output");

    input->GetOutputSlot(0).Connect(conv2d->GetInputSlot(0));
    conv2d->GetOutputSlot(0).Connect(activation->GetInputSlot(0));
    activation->GetOutputSlot(0).Connect(output->GetInputSlot(0));

    input->GetOutputSlot(0).SetTensorInfo(inputInfo);
    conv2d->GetOutputSlot(0).SetTensorInfo(outputInfo);
    activation->GetOutputSlot(0).SetTensorInfo(outputInfo);

    // optimize the network
    std::vector<armnn::BackendId> backends = { backendId };
    IOptimizedNetworkPtr optNet = Optimize(*net, backends, runtime.GetDeviceSpec());

    ProfilingGuid optNetGuid = optNet->GetGuid();

    // Load it into the runtime. It should success.
    armnn::NetworkId netId;
    BOOST_TEST(runtime.LoadNetwork(netId, std::move(optNet)) == Status::Success);

    profiling::ProfilingServiceRuntimeHelper profilingServiceHelper(GetProfilingService(&runtime));
    profiling::BufferManager& bufferManager = profilingServiceHelper.GetProfilingBufferManager();
    auto readableBuffer = bufferManager.GetReadableBuffer();

    // Profiling is enable, the post-optimisation structure should be created
    BOOST_CHECK(readableBuffer != nullptr);

    unsigned int size = readableBuffer->GetSize();
    BOOST_CHECK(size == 1556);

    const unsigned char* readableData = readableBuffer->GetReadableData();
    BOOST_CHECK(readableData != nullptr);

    unsigned int offset = 0;

    // Verify Header
    VerifyTimelineHeaderBinary(readableData, offset, 1548);

    // Post-optimisation network
    // Network entity
    VerifyTimelineEntityBinaryPacketData(optNetGuid, readableData, offset);

    // Entity - Type relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::LabelLink,
                                               EmptyOptional(),
                                               optNetGuid,
                                               EmptyOptional(),
                                               readableData,
                                               offset);

    // Type label relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::LabelLink,
                                               EmptyOptional(),
                                               EmptyOptional(),
                                               LabelsAndEventClasses::TYPE_GUID,
                                               readableData,
                                               offset);

    // Input layer
    // Input layer entity
    VerifyTimelineEntityBinaryPacketData(input->GetGuid(), readableData, offset);

    // Name Entity
    VerifyTimelineLabelBinaryPacketData(EmptyOptional(), "input", readableData, offset);

    // Entity - Name relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::LabelLink,
                                               EmptyOptional(),
                                               input->GetGuid(),
                                               EmptyOptional(),
                                               readableData,
                                               offset);

    // Name label relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::LabelLink,
                                               EmptyOptional(),
                                               EmptyOptional(),
                                               LabelsAndEventClasses::NAME_GUID,
                                               readableData,
                                               offset);

    // Entity - Type relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::LabelLink,
                                               EmptyOptional(),
                                               input->GetGuid(),
                                               EmptyOptional(),
                                               readableData,
                                               offset);

    // Type label relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::LabelLink,
                                               EmptyOptional(),
                                               EmptyOptional(),
                                               LabelsAndEventClasses::TYPE_GUID,
                                               readableData,
                                               offset);

    // Network - Input layer relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::RetentionLink,
                                               EmptyOptional(),
                                               optNetGuid,
                                               input->GetGuid(),
                                               readableData,
                                               offset);

    // Conv2d layer
    // Conv2d layer entity
    VerifyTimelineEntityBinaryPacketData(conv2d->GetGuid(), readableData, offset);

    // Name entity
    VerifyTimelineLabelBinaryPacketData(EmptyOptional(), "<Unnamed>", readableData, offset);

    // Entity - Name relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::LabelLink,
                                               EmptyOptional(),
                                               conv2d->GetGuid(),
                                               EmptyOptional(),
                                               readableData,
                                               offset);

    // Name label relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::LabelLink,
                                               EmptyOptional(),
                                               EmptyOptional(),
                                               LabelsAndEventClasses::NAME_GUID,
                                               readableData,
                                               offset);

    // Entity - Type relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::LabelLink,
                                               EmptyOptional(),
                                               conv2d->GetGuid(),
                                               EmptyOptional(),
                                               readableData,
                                               offset);

    // Type label relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::LabelLink,
                                               EmptyOptional(),
                                               EmptyOptional(),
                                               LabelsAndEventClasses::TYPE_GUID,
                                               readableData,
                                               offset);

    // Network - Conv2d layer relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::RetentionLink,
                                               EmptyOptional(),
                                               optNetGuid,
                                               conv2d->GetGuid(),
                                               readableData,
                                               offset);

    // Input layer - Conv2d layer relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::RetentionLink,
                                               EmptyOptional(),
                                               input->GetGuid(),
                                               conv2d->GetGuid(),
                                               readableData,
                                               offset);

    // Entity - Type relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::LabelLink,
                                               EmptyOptional(),
                                               EmptyOptional(),
                                               LabelsAndEventClasses::CONNECTION_GUID,
                                               readableData,
                                               offset);

    // Type label relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::LabelLink,
                                               EmptyOptional(),
                                               EmptyOptional(),
                                               LabelsAndEventClasses::TYPE_GUID,
                                               readableData,
                                               offset);

    // Conv2d workload
    // Conv2d workload entity
    VerifyTimelineEntityBinaryPacketData(EmptyOptional(), readableData, offset);

    // Entity - Type relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::LabelLink,
                                               EmptyOptional(),
                                               EmptyOptional(),
                                               EmptyOptional(),
                                               readableData,
                                               offset);

    // Type label relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::LabelLink,
                                               EmptyOptional(),
                                               EmptyOptional(),
                                               LabelsAndEventClasses::TYPE_GUID,
                                               readableData,
                                               offset);

    // BackendId entity
    VerifyTimelineLabelBinaryPacketData(EmptyOptional(), backendId.Get(), readableData, offset);

    // Entity - BackendId relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::LabelLink,
                                               EmptyOptional(),
                                               EmptyOptional(),
                                               EmptyOptional(),
                                               readableData,
                                               offset);

    // BackendId label relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::LabelLink,
                                               EmptyOptional(),
                                               EmptyOptional(),
                                               LabelsAndEventClasses::BACKENDID_GUID,
                                               readableData,
                                               offset);

    // Conv2d layer - Conv2d workload relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::RetentionLink,
                                               EmptyOptional(),
                                               conv2d->GetGuid(),
                                               EmptyOptional(),
                                               readableData,
                                               offset);

    // Activation layer
    // Activation layer entity
    VerifyTimelineEntityBinaryPacketData(activation->GetGuid(), readableData, offset);

    // Name entity
    VerifyTimelineLabelBinaryPacketData(EmptyOptional(), "activation", readableData, offset);

    // Entity - Name relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::LabelLink,
                                               EmptyOptional(),
                                               activation->GetGuid(),
                                               EmptyOptional(),
                                               readableData,
                                               offset);

    // Name label relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::LabelLink,
                                               EmptyOptional(),
                                               EmptyOptional(),
                                               LabelsAndEventClasses::NAME_GUID,
                                               readableData,
                                               offset);

    // Entity - Type relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::LabelLink,
                                               EmptyOptional(),
                                               activation->GetGuid(),
                                               EmptyOptional(),
                                               readableData,
                                               offset);

    // Type label relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::LabelLink,
                                               EmptyOptional(),
                                               EmptyOptional(),
                                               LabelsAndEventClasses::TYPE_GUID,
                                               readableData,
                                               offset);

    // Network - Activation layer relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::RetentionLink,
                                               EmptyOptional(),
                                               optNetGuid,
                                               activation->GetGuid(),
                                               readableData,
                                               offset);

    // Conv2d layer - Activation layer relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::RetentionLink,
                                               EmptyOptional(),
                                               conv2d->GetGuid(),
                                               activation->GetGuid(),
                                               readableData,
                                               offset);

    // Entity - Type relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::LabelLink,
                                               EmptyOptional(),
                                               EmptyOptional(),
                                               LabelsAndEventClasses::CONNECTION_GUID,
                                               readableData,
                                               offset);

    // Type label relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::LabelLink,
                                               EmptyOptional(),
                                               EmptyOptional(),
                                               LabelsAndEventClasses::TYPE_GUID,
                                               readableData,
                                               offset);

    // Activation workload
    // Activation workload entity
    VerifyTimelineEntityBinaryPacketData(EmptyOptional(), readableData, offset);

    // Entity - Type relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::LabelLink,
                                               EmptyOptional(),
                                               EmptyOptional(),
                                               EmptyOptional(),
                                               readableData,
                                               offset);

    // Type label relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::LabelLink,
                                               EmptyOptional(),
                                               EmptyOptional(),
                                               LabelsAndEventClasses::TYPE_GUID,
                                               readableData,
                                               offset);

    // BackendId entity
    VerifyTimelineLabelBinaryPacketData(EmptyOptional(), backendId.Get(), readableData, offset);

    // Entity - BackendId relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::LabelLink,
                                               EmptyOptional(),
                                               EmptyOptional(),
                                               EmptyOptional(),
                                               readableData,
                                               offset);

    // BackendId label relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::LabelLink,
                                               EmptyOptional(),
                                               EmptyOptional(),
                                               LabelsAndEventClasses::BACKENDID_GUID,
                                               readableData,
                                               offset);

    // Activation layer - Activation workload relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::RetentionLink,
                                               EmptyOptional(),
                                               activation->GetGuid(),
                                               EmptyOptional(),
                                               readableData,
                                               offset);

    // Output layer
    // Output layer entity
    VerifyTimelineEntityBinaryPacketData(output->GetGuid(), readableData, offset);

    // Name entity
    VerifyTimelineLabelBinaryPacketData(EmptyOptional(), "output", readableData, offset);

    // Entity - Name relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::LabelLink,
                                               EmptyOptional(),
                                               output->GetGuid(),
                                               EmptyOptional(),
                                               readableData,
                                               offset);

    // Name label relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::LabelLink,
                                               EmptyOptional(),
                                               EmptyOptional(),
                                               LabelsAndEventClasses::NAME_GUID,
                                               readableData,
                                               offset);

    // Entity - Type relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::LabelLink,
                                               EmptyOptional(),
                                               output->GetGuid(),
                                               EmptyOptional(),
                                               readableData,
                                               offset);

    // Type label relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::LabelLink,
                                               EmptyOptional(),
                                               EmptyOptional(),
                                               LabelsAndEventClasses::TYPE_GUID,
                                               readableData,
                                               offset);

    // Network - Output layer relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::RetentionLink,
                                               EmptyOptional(),
                                               optNetGuid,
                                               output->GetGuid(),
                                               readableData,
                                               offset);

    // Activation layer - Output layer relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::RetentionLink,
                                               EmptyOptional(),
                                               activation->GetGuid(),
                                               output->GetGuid(),
                                               readableData,
                                               offset);

    // Entity - Type relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::LabelLink,
                                               EmptyOptional(),
                                               EmptyOptional(),
                                               LabelsAndEventClasses::CONNECTION_GUID,
                                               readableData,
                                               offset);

    // Type label relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::LabelLink,
                                               EmptyOptional(),
                                               EmptyOptional(),
                                               LabelsAndEventClasses::TYPE_GUID,
                                               readableData,
                                               offset);

    bufferManager.MarkRead(readableBuffer);

    // Creates structures for input & output.
    std::vector<float> inputData(inputInfo.GetNumElements());
    std::vector<float> outputData(outputInfo.GetNumElements());

    InputTensors inputTensors
        {
        {0, ConstTensor(runtime.GetInputTensorInfo(netId, 0), inputData.data())}
        };
    OutputTensors outputTensors
        {
        {0, Tensor(runtime.GetOutputTensorInfo(netId, 0), outputData.data())}
        };

    // Does the inference.
    runtime.EnqueueWorkload(netId, inputTensors, outputTensors);

    // Get readable buffer for inference timeline
    auto inferenceReadableBuffer = bufferManager.GetReadableBuffer();
    BOOST_CHECK(inferenceReadableBuffer != nullptr);

    // Get readable buffer for output workload
    auto outputReadableBuffer = bufferManager.GetReadableBuffer();
    BOOST_CHECK(outputReadableBuffer != nullptr);

    // Get readable buffer for input workload
    auto inputReadableBuffer = bufferManager.GetReadableBuffer();
    BOOST_CHECK(inputReadableBuffer != nullptr);

    // Validate input workload data
    size = inputReadableBuffer->GetSize();
    BOOST_CHECK(size == 204);

    readableData = inputReadableBuffer->GetReadableData();
    BOOST_CHECK(readableData != nullptr);

    offset = 0;

    // Verify Header
    VerifyTimelineHeaderBinary(readableData, offset, 196);

    // Input workload
    // Input workload entity
    VerifyTimelineEntityBinaryPacketData(EmptyOptional(), readableData, offset);

    // Entity - Type relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::LabelLink,
                                               EmptyOptional(),
                                               EmptyOptional(),
                                               EmptyOptional(),
                                               readableData,
                                               offset);

    // Type label relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::LabelLink,
                                               EmptyOptional(),
                                               EmptyOptional(),
                                               LabelsAndEventClasses::TYPE_GUID,
                                               readableData,
                                               offset);

    // BackendId entity
    VerifyTimelineLabelBinaryPacketData(EmptyOptional(), backendId.Get(), readableData, offset);

    // Entity - BackendId relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::LabelLink,
                                               EmptyOptional(),
                                               EmptyOptional(),
                                               EmptyOptional(),
                                               readableData,
                                               offset);

    // BackendId label relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::LabelLink,
                                               EmptyOptional(),
                                               EmptyOptional(),
                                               LabelsAndEventClasses::BACKENDID_GUID,
                                               readableData,
                                               offset);

    // Input layer - Input workload relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::RetentionLink,
                                               EmptyOptional(),
                                               input->GetGuid(),
                                               EmptyOptional(),
                                               readableData,
                                               offset);

    bufferManager.MarkRead(inputReadableBuffer);

    // Validate output workload data
    size = outputReadableBuffer->GetSize();
    BOOST_CHECK(size == 204);

    readableData = outputReadableBuffer->GetReadableData();
    BOOST_CHECK(readableData != nullptr);

    offset = 0;

    // Verify Header
    VerifyTimelineHeaderBinary(readableData, offset, 196);

    // Output workload
    // Output workload entity
    VerifyTimelineEntityBinaryPacketData(EmptyOptional(), readableData, offset);

    // Entity - Type relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::LabelLink,
                                               EmptyOptional(),
                                               EmptyOptional(),
                                               EmptyOptional(),
                                               readableData,
                                               offset);

    // Type label relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::LabelLink,
                                               EmptyOptional(),
                                               EmptyOptional(),
                                               LabelsAndEventClasses::TYPE_GUID,
                                               readableData,
                                               offset);

    // BackendId entity
    VerifyTimelineLabelBinaryPacketData(EmptyOptional(), backendId.Get(), readableData, offset);

    // Entity - BackendId relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::LabelLink,
                                               EmptyOptional(),
                                               EmptyOptional(),
                                               EmptyOptional(),
                                               readableData,
                                               offset);

    // BackendId label relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::LabelLink,
                                               EmptyOptional(),
                                               EmptyOptional(),
                                               LabelsAndEventClasses::BACKENDID_GUID,
                                               readableData,
                                               offset);

    // Output layer - Output workload relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::RetentionLink,
                                               EmptyOptional(),
                                               output->GetGuid(),
                                               EmptyOptional(),
                                               readableData,
                                               offset);

    bufferManager.MarkRead(outputReadableBuffer);

    // Validate inference data
    size = inferenceReadableBuffer->GetSize();
    BOOST_CHECK(size == 1596);

    readableData = inferenceReadableBuffer->GetReadableData();
    BOOST_CHECK(readableData != nullptr);

    offset = 0;

    // Verify Header
    VerifyTimelineHeaderBinary(readableData, offset, 1588);

    // Inference timeline trace
    // Inference entity
    VerifyTimelineEntityBinaryPacketData(EmptyOptional(), readableData, offset);

    // Entity - Type relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::LabelLink,
                                               EmptyOptional(),
                                               EmptyOptional(),
                                               LabelsAndEventClasses::INFERENCE_GUID,
                                               readableData,
                                               offset);

    // Type label relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::LabelLink,
                                               EmptyOptional(),
                                               EmptyOptional(),
                                               LabelsAndEventClasses::TYPE_GUID,
                                               readableData,
                                               offset);

    // Network - Inference relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::RetentionLink,
                                               EmptyOptional(),
                                               optNetGuid,
                                               EmptyOptional(),
                                               readableData,
                                               offset);

    // Start Inference life
    // Event packet - timeline, threadId, eventGuid
    VerifyTimelineEventBinaryPacket(EmptyOptional(), EmptyOptional(), EmptyOptional(), readableData, offset);

    // Inference - event relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::ExecutionLink,
                                               EmptyOptional(),
                                               EmptyOptional(),
                                               EmptyOptional(),
                                               readableData,
                                               offset);

    // Event - event class relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::DataLink,
                                               EmptyOptional(),
                                               EmptyOptional(),
                                               LabelsAndEventClasses::ARMNN_PROFILING_SOL_EVENT_CLASS,
                                               readableData,
                                               offset);

    // Execution
    // Input workload execution
    // Input workload execution entity
    VerifyTimelineEntityBinaryPacketData(EmptyOptional(), readableData, offset);

    // Entity - Type relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::LabelLink,
                                               EmptyOptional(),
                                               EmptyOptional(),
                                               LabelsAndEventClasses::WORKLOAD_EXECUTION_GUID,
                                               readableData,
                                               offset);

    // Type label relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::LabelLink,
                                               EmptyOptional(),
                                               EmptyOptional(),
                                               LabelsAndEventClasses::TYPE_GUID,
                                               readableData,
                                               offset);

    // Inference - Workload execution relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::RetentionLink,
                                               EmptyOptional(),
                                               EmptyOptional(),
                                               EmptyOptional(),
                                               readableData,
                                               offset);

    // Workload - Workload execution relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::RetentionLink,
                                               EmptyOptional(),
                                               EmptyOptional(),
                                               EmptyOptional(),
                                               readableData,
                                               offset);

    // Start Input workload execution life
    // Event packet - timeline, threadId, eventGuid
    VerifyTimelineEventBinaryPacket(EmptyOptional(), EmptyOptional(), EmptyOptional(), readableData, offset);

    // Input workload execution - event relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::ExecutionLink,
                                               EmptyOptional(),
                                               EmptyOptional(),
                                               EmptyOptional(),
                                               readableData,
                                               offset);

    // Event - event class relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::DataLink,
                                               EmptyOptional(),
                                               EmptyOptional(),
                                               LabelsAndEventClasses::ARMNN_PROFILING_SOL_EVENT_CLASS,
                                               readableData,
                                               offset);

    // End of Input workload execution life
    // Event packet - timeline, threadId, eventGuid
    VerifyTimelineEventBinaryPacket(EmptyOptional(), EmptyOptional(), EmptyOptional(), readableData, offset);

    // Input workload execution - event relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::ExecutionLink,
                                               EmptyOptional(),
                                               EmptyOptional(),
                                               EmptyOptional(),
                                               readableData,
                                               offset);

    // Event - event class relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::DataLink,
                                               EmptyOptional(),
                                               EmptyOptional(),
                                               LabelsAndEventClasses::ARMNN_PROFILING_EOL_EVENT_CLASS,
                                               readableData,
                                               offset);

    // Conv2d workload execution
    // Conv2d workload execution entity
    VerifyTimelineEntityBinaryPacketData(EmptyOptional(), readableData, offset);

    // Entity - Type relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::LabelLink,
                                               EmptyOptional(),
                                               EmptyOptional(),
                                               LabelsAndEventClasses::WORKLOAD_EXECUTION_GUID,
                                               readableData,
                                               offset);

    // Type label relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::LabelLink,
                                               EmptyOptional(),
                                               EmptyOptional(),
                                               LabelsAndEventClasses::TYPE_GUID,
                                               readableData,
                                               offset);

    // Inference - Workload execution relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::RetentionLink,
                                               EmptyOptional(),
                                               EmptyOptional(),
                                               EmptyOptional(),
                                               readableData,
                                               offset);

    // Workload - Workload execution relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::RetentionLink,
                                               EmptyOptional(),
                                               EmptyOptional(),
                                               EmptyOptional(),
                                               readableData,
                                               offset);

    // Start Conv2d workload execution life
    // Event packet - timeline, threadId, eventGuid
    VerifyTimelineEventBinaryPacket(EmptyOptional(), EmptyOptional(), EmptyOptional(), readableData, offset);

    // Conv2d workload execution - event relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::ExecutionLink,
                                               EmptyOptional(),
                                               EmptyOptional(),
                                               EmptyOptional(),
                                               readableData,
                                               offset);

    // Event - event class relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::DataLink,
                                               EmptyOptional(),
                                               EmptyOptional(),
                                               LabelsAndEventClasses::ARMNN_PROFILING_SOL_EVENT_CLASS,
                                               readableData,
                                               offset);

    // End of Conv2d workload execution life
    // Event packet - timeline, threadId, eventGuid
    VerifyTimelineEventBinaryPacket(EmptyOptional(), EmptyOptional(), EmptyOptional(), readableData, offset);

    // Conv2d workload execution - event relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::ExecutionLink,
                                               EmptyOptional(),
                                               EmptyOptional(),
                                               EmptyOptional(),
                                               readableData,
                                               offset);

    // Event - event class relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::DataLink,
                                               EmptyOptional(),
                                               EmptyOptional(),
                                               LabelsAndEventClasses::ARMNN_PROFILING_EOL_EVENT_CLASS,
                                               readableData,
                                               offset);

    // Activation workload execution
    // Activation workload execution entity
    VerifyTimelineEntityBinaryPacketData(EmptyOptional(), readableData, offset);

    // Entity - Type relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::LabelLink,
                                               EmptyOptional(),
                                               EmptyOptional(),
                                               LabelsAndEventClasses::WORKLOAD_EXECUTION_GUID,
                                               readableData,
                                               offset);

    // Type label relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::LabelLink,
                                               EmptyOptional(),
                                               EmptyOptional(),
                                               LabelsAndEventClasses::TYPE_GUID,
                                               readableData,
                                               offset);

    // Inference - Workload execution relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::RetentionLink,
                                               EmptyOptional(),
                                               EmptyOptional(),
                                               EmptyOptional(),
                                               readableData,
                                               offset);

    // Workload - Workload execution relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::RetentionLink,
                                               EmptyOptional(),
                                               EmptyOptional(),
                                               EmptyOptional(),
                                               readableData,
                                               offset);

    // Start Activation workload execution life
    // Event packet - timeline, threadId, eventGuid
    VerifyTimelineEventBinaryPacket(EmptyOptional(), EmptyOptional(), EmptyOptional(), readableData, offset);

    // Activation workload execution - event relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::ExecutionLink,
                                               EmptyOptional(),
                                               EmptyOptional(),
                                               EmptyOptional(),
                                               readableData,
                                               offset);

    // Event - event class relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::DataLink,
                                               EmptyOptional(),
                                               EmptyOptional(),
                                               LabelsAndEventClasses::ARMNN_PROFILING_SOL_EVENT_CLASS,
                                               readableData,
                                               offset);

    // End of Activation workload execution life
    // Event packet - timeline, threadId, eventGuid
    VerifyTimelineEventBinaryPacket(EmptyOptional(), EmptyOptional(), EmptyOptional(), readableData, offset);

    // Activation workload execution - event relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::ExecutionLink,
                                               EmptyOptional(),
                                               EmptyOptional(),
                                               EmptyOptional(),
                                               readableData,
                                               offset);

    // Event - event class relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::DataLink,
                                               EmptyOptional(),
                                               EmptyOptional(),
                                               LabelsAndEventClasses::ARMNN_PROFILING_EOL_EVENT_CLASS,
                                               readableData,
                                               offset);

    // Output workload execution
    // Output workload execution entity
    VerifyTimelineEntityBinaryPacketData(EmptyOptional(), readableData, offset);

    // Entity - Type relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::LabelLink,
                                               EmptyOptional(),
                                               EmptyOptional(),
                                               LabelsAndEventClasses::WORKLOAD_EXECUTION_GUID,
                                               readableData,
                                               offset);

    // Type label relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::LabelLink,
                                               EmptyOptional(),
                                               EmptyOptional(),
                                               LabelsAndEventClasses::TYPE_GUID,
                                               readableData,
                                               offset);

    // Inference - Workload execution relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::RetentionLink,
                                               EmptyOptional(),
                                               EmptyOptional(),
                                               EmptyOptional(),
                                               readableData,
                                               offset);

    // Workload - Workload execution relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::RetentionLink,
                                               EmptyOptional(),
                                               EmptyOptional(),
                                               EmptyOptional(),
                                               readableData,
                                               offset);

    // Start Output workload execution life
    // Event packet - timeline, threadId, eventGuid
    VerifyTimelineEventBinaryPacket(EmptyOptional(), EmptyOptional(), EmptyOptional(), readableData, offset);

    // Output workload execution - event relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::ExecutionLink,
                                               EmptyOptional(),
                                               EmptyOptional(),
                                               EmptyOptional(),
                                               readableData,
                                               offset);

    // Event - event class relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::DataLink,
                                               EmptyOptional(),
                                               EmptyOptional(),
                                               LabelsAndEventClasses::ARMNN_PROFILING_SOL_EVENT_CLASS,
                                               readableData,
                                               offset);

    // End of Normalize workload execution life
    // Event packet - timeline, threadId, eventGuid
    VerifyTimelineEventBinaryPacket(EmptyOptional(), EmptyOptional(), EmptyOptional(), readableData, offset);

    // Output workload execution - event relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::ExecutionLink,
                                               EmptyOptional(),
                                               EmptyOptional(),
                                               EmptyOptional(),
                                               readableData,
                                               offset);

    // Event - event class relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::DataLink,
                                               EmptyOptional(),
                                               EmptyOptional(),
                                               LabelsAndEventClasses::ARMNN_PROFILING_EOL_EVENT_CLASS,
                                               readableData,
                                               offset);

    // End of Inference life
    // Event packet - timeline, threadId, eventGuid
    VerifyTimelineEventBinaryPacket(EmptyOptional(), EmptyOptional(), EmptyOptional(), readableData, offset);

    // Inference - event relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::ExecutionLink,
                                               EmptyOptional(),
                                               EmptyOptional(),
                                               EmptyOptional(),
                                               readableData,
                                               offset);

    // Event - event class relationship
    VerifyTimelineRelationshipBinaryPacketData(ProfilingRelationshipType::DataLink,
                                               EmptyOptional(),
                                               EmptyOptional(),
                                               LabelsAndEventClasses::ARMNN_PROFILING_EOL_EVENT_CLASS,
                                               readableData,
                                               offset);

    bufferManager.MarkRead(inferenceReadableBuffer);
}
