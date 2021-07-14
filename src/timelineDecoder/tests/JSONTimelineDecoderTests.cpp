//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <JSONTimelineDecoder.hpp>
#include <server/include/timelineDecoder/TimelineCaptureCommandHandler.hpp>
#include <server/include/timelineDecoder/TimelineDecoder.hpp>
#include <armnnUtils/Filesystem.hpp>

#include <doctest/doctest.h>

#include <fstream>

TEST_SUITE("JSONTimelineDecoderTests")
{
using namespace armnn;
using namespace timelinedecoder;

void RunSimpleModelThroughDecoder(JSONTimelineDecoder& timelineDecoder)
{
    /*
    * Building Simple model structure to test
    */
    arm::pipe::ITimelineDecoder::Label name;
    name.m_Guid = uint64_t(10420554295983197538U);
    name.m_Name = "name";
    timelineDecoder.CreateLabel(name);

    arm::pipe::ITimelineDecoder::Label type;
    type.m_Guid = uint64_t(14196220359693045352U);
    type.m_Name = "type";
    timelineDecoder.CreateLabel(type);

    arm::pipe::ITimelineDecoder::Label index;
    index.m_Guid = uint64_t(13922236767355949814U);
    index.m_Name = "index";
    timelineDecoder.CreateLabel(index);

    arm::pipe::ITimelineDecoder::Label backendId;
    backendId.m_Guid = uint64_t(10874037804557439415U);
    backendId.m_Name = "backendId";
    timelineDecoder.CreateLabel(backendId);

    arm::pipe::ITimelineDecoder::Label layer;
    layer.m_Guid = uint64_t(14761340794127440397U);
    layer.m_Name = "layer";
    timelineDecoder.CreateLabel(layer);

    arm::pipe::ITimelineDecoder::Label workload;
    workload.m_Guid = uint64_t(15704252740552608110U);
    workload.m_Name = "workload";
    timelineDecoder.CreateLabel(workload);

    arm::pipe::ITimelineDecoder::Label network;
    network.m_Guid = uint64_t(16862199137063532871U);
    network.m_Name = "network";
    timelineDecoder.CreateLabel(network);

    arm::pipe::ITimelineDecoder::Label connection;
    connection.m_Guid = uint64_t(15733717748792475675U);
    connection.m_Name = "connection";
    timelineDecoder.CreateLabel(connection);

    arm::pipe::ITimelineDecoder::Label inference;
    inference.m_Guid = uint64_t(15026600058430441282U);
    inference.m_Name = "inference";
    timelineDecoder.CreateLabel(inference);

    arm::pipe::ITimelineDecoder::Label workload_execution;
    workload_execution.m_Guid = uint64_t(10172155312650606003U);
    workload_execution.m_Name = "workload_execution";
    timelineDecoder.CreateLabel(workload_execution);

    arm::pipe::ITimelineDecoder::EventClass eventClass1;
    eventClass1.m_Guid = uint64_t(17170418158534996719U);
    timelineDecoder.CreateEventClass(eventClass1);

    arm::pipe::ITimelineDecoder::EventClass eventClass2;
    eventClass2.m_Guid = uint64_t(10812061579584851344U);
    timelineDecoder.CreateEventClass(eventClass2);

    arm::pipe::ITimelineDecoder::Entity entity6;
    entity6.m_Guid = uint64_t(6);
    timelineDecoder.CreateEntity(entity6);

    arm::pipe::ITimelineDecoder::Relationship relationship7;
    relationship7.m_Guid = uint64_t(7);
    relationship7.m_RelationshipType = arm::pipe::ITimelineDecoder::RelationshipType::LabelLink;
    relationship7.m_HeadGuid = uint64_t(6);
    relationship7.m_TailGuid = uint64_t(16862199137063532871U);
    timelineDecoder.CreateRelationship(relationship7);

    arm::pipe::ITimelineDecoder::Relationship relationship8;
    relationship8.m_Guid = uint64_t(8);
    relationship8.m_RelationshipType = arm::pipe::ITimelineDecoder::RelationshipType::LabelLink;
    relationship8.m_HeadGuid = uint64_t(7);
    relationship8.m_TailGuid = uint64_t(14196220359693045352U);
    timelineDecoder.CreateRelationship(relationship8);

    // Adding normalization layer
    arm::pipe::ITimelineDecoder::Entity entity0;
    entity0.m_Guid = uint64_t(0);
    timelineDecoder.CreateEntity(entity0);

    arm::pipe::ITimelineDecoder::Label input;
    input.m_Guid = uint64_t(18179123836411086572U);
    input.m_Name = "input";
    timelineDecoder.CreateLabel(input);

    arm::pipe::ITimelineDecoder::Relationship relationship9;
    relationship9.m_Guid = uint64_t(9);
    relationship9.m_RelationshipType = arm::pipe::ITimelineDecoder::RelationshipType::LabelLink;
    relationship9.m_HeadGuid = uint64_t(0);
    relationship9.m_TailGuid = uint64_t(18179123836411086572U);
    timelineDecoder.CreateRelationship(relationship9);

    arm::pipe::ITimelineDecoder::Relationship relationship10;
    relationship10.m_Guid = uint64_t(10);
    relationship10.m_RelationshipType = arm::pipe::ITimelineDecoder::RelationshipType::LabelLink;
    relationship10.m_HeadGuid = uint64_t(9);
    relationship10.m_TailGuid = uint64_t(10420554295983197538U);
    timelineDecoder.CreateRelationship(relationship10);

    arm::pipe::ITimelineDecoder::Relationship relationship11;
    relationship11.m_Guid = uint64_t(11);
    relationship11.m_RelationshipType = arm::pipe::ITimelineDecoder::RelationshipType::LabelLink;
    relationship11.m_HeadGuid = uint64_t(0);
    relationship11.m_TailGuid = uint64_t(14761340794127440397U);
    timelineDecoder.CreateRelationship(relationship11);

    arm::pipe::ITimelineDecoder::Relationship relationship12;
    relationship12.m_Guid = uint64_t(12);
    relationship12.m_RelationshipType = arm::pipe::ITimelineDecoder::RelationshipType::LabelLink;
    relationship12.m_HeadGuid = uint64_t(11);
    relationship12.m_TailGuid = uint64_t(14196220359693045352U);
    timelineDecoder.CreateRelationship(relationship12);

    arm::pipe::ITimelineDecoder::Relationship relationship13;
    relationship13.m_Guid = uint64_t(13);
    relationship13.m_RelationshipType = arm::pipe::ITimelineDecoder::RelationshipType::RetentionLink;
    relationship13.m_HeadGuid = uint64_t(6);
    relationship13.m_TailGuid = uint64_t(0);
    timelineDecoder.CreateRelationship(relationship13);


    // Adding normalization layer
    arm::pipe::ITimelineDecoder::Entity entity1;
    entity1.m_Guid = uint64_t(1);
    timelineDecoder.CreateEntity(entity1);

    arm::pipe::ITimelineDecoder::Label normalization;
    normalization.m_Guid = uint64_t(15955949569988957863U);
    normalization.m_Name = "normalization";
    timelineDecoder.CreateLabel(normalization);

    arm::pipe::ITimelineDecoder::Relationship relationship14;
    relationship14.m_Guid = uint64_t(14);
    relationship14.m_RelationshipType = arm::pipe::ITimelineDecoder::RelationshipType::LabelLink;
    relationship14.m_HeadGuid = uint64_t(1);
    relationship14.m_TailGuid = uint64_t(15955949569988957863U);
    timelineDecoder.CreateRelationship(relationship14);

    arm::pipe::ITimelineDecoder::Relationship relationship15;
    relationship15.m_Guid = uint64_t(15);
    relationship15.m_RelationshipType = arm::pipe::ITimelineDecoder::RelationshipType::LabelLink;
    relationship15.m_HeadGuid = uint64_t(14);
    relationship15.m_TailGuid = uint64_t(10420554295983197538U);
    timelineDecoder.CreateRelationship(relationship15);

    arm::pipe::ITimelineDecoder::Relationship relationship16;
    relationship16.m_Guid = uint64_t(16);
    relationship16.m_RelationshipType = arm::pipe::ITimelineDecoder::RelationshipType::LabelLink;
    relationship16.m_HeadGuid = uint64_t(1);
    relationship16.m_TailGuid = uint64_t(14761340794127440397U);
    timelineDecoder.CreateRelationship(relationship16);

    arm::pipe::ITimelineDecoder::Relationship relationship17;
    relationship17.m_Guid = uint64_t(17);
    relationship17.m_RelationshipType = arm::pipe::ITimelineDecoder::RelationshipType::LabelLink;
    relationship17.m_HeadGuid = uint64_t(16);
    relationship17.m_TailGuid = uint64_t(14196220359693045352U);
    timelineDecoder.CreateRelationship(relationship17);

    arm::pipe::ITimelineDecoder::Relationship relationship18;
    relationship18.m_Guid = uint64_t(18);
    relationship18.m_RelationshipType = arm::pipe::ITimelineDecoder::RelationshipType::RetentionLink;
    relationship18.m_HeadGuid = uint64_t(6);
    relationship18.m_TailGuid = uint64_t(1);
    timelineDecoder.CreateRelationship(relationship18);

    arm::pipe::ITimelineDecoder::Relationship relationship19;
    relationship19.m_Guid = uint64_t(19);
    relationship19.m_RelationshipType = arm::pipe::ITimelineDecoder::RelationshipType::RetentionLink;
    relationship19.m_HeadGuid = uint64_t(0);
    relationship19.m_TailGuid = uint64_t(1);
    timelineDecoder.CreateRelationship(relationship19);

    arm::pipe::ITimelineDecoder::Relationship relationship20;
    relationship20.m_Guid = uint64_t(20);
    relationship20.m_RelationshipType = arm::pipe::ITimelineDecoder::RelationshipType::LabelLink;
    relationship20.m_HeadGuid = uint64_t(19);
    relationship20.m_TailGuid = uint64_t(15733717748792475675U);
    timelineDecoder.CreateRelationship(relationship20);

    arm::pipe::ITimelineDecoder::Relationship relationship21;
    relationship21.m_Guid = uint64_t(21);
    relationship21.m_RelationshipType = arm::pipe::ITimelineDecoder::RelationshipType::LabelLink;
    relationship21.m_HeadGuid = uint64_t(20);
    relationship21.m_TailGuid = uint64_t(14196220359693045352U);
    timelineDecoder.CreateRelationship(relationship21);


    arm::pipe::ITimelineDecoder::Entity entity22;
    entity22.m_Guid = uint64_t(22);
    timelineDecoder.CreateEntity(entity22);

    arm::pipe::ITimelineDecoder::Relationship relationship23;
    relationship23.m_Guid = uint64_t(23);
    relationship23.m_RelationshipType = arm::pipe::ITimelineDecoder::RelationshipType::LabelLink;
    relationship23.m_HeadGuid = uint64_t(22);
    relationship23.m_TailGuid = uint64_t(15704252740552608110U);
    timelineDecoder.CreateRelationship(relationship23);

    arm::pipe::ITimelineDecoder::Relationship relationship24;
    relationship24.m_Guid = uint64_t(24);
    relationship24.m_RelationshipType = arm::pipe::ITimelineDecoder::RelationshipType::LabelLink;
    relationship24.m_HeadGuid = uint64_t(23);
    relationship24.m_TailGuid = uint64_t(14196220359693045352U);
    timelineDecoder.CreateRelationship(relationship24);


    arm::pipe::ITimelineDecoder::Label CpuRef;
    CpuRef.m_Guid = uint64_t(9690680943817437852U);
    CpuRef.m_Name = "CpuRef";
    timelineDecoder.CreateLabel(CpuRef);


    arm::pipe::ITimelineDecoder::Relationship relationship25;
    relationship25.m_Guid = uint64_t(25);
    relationship25.m_RelationshipType = arm::pipe::ITimelineDecoder::RelationshipType::LabelLink;
    relationship25.m_HeadGuid = uint64_t(22);
    relationship25.m_TailGuid = uint64_t(9690680943817437852U);
    timelineDecoder.CreateRelationship(relationship25);

    arm::pipe::ITimelineDecoder::Relationship relationship26;
    relationship26.m_Guid = uint64_t(26);
    relationship26.m_RelationshipType = arm::pipe::ITimelineDecoder::RelationshipType::LabelLink;
    relationship26.m_HeadGuid = uint64_t(25);
    relationship26.m_TailGuid = uint64_t(10874037804557439415U);
    timelineDecoder.CreateRelationship(relationship26);

    arm::pipe::ITimelineDecoder::Relationship relationship27;
    relationship27.m_Guid = uint64_t(27);
    relationship27.m_RelationshipType = arm::pipe::ITimelineDecoder::RelationshipType::RetentionLink   ;
    relationship27.m_HeadGuid = uint64_t(1);
    relationship27.m_TailGuid = uint64_t(22);
    timelineDecoder.CreateRelationship(relationship27);

    // Adding output layer
    arm::pipe::ITimelineDecoder::Entity entity2;
    entity2.m_Guid = uint64_t(2);
    timelineDecoder.CreateEntity(entity2);

    arm::pipe::ITimelineDecoder::Label output;
    output.m_Guid = uint64_t(18419179028513879730U);
    output.m_Name = "output";
    timelineDecoder.CreateLabel(output);

    arm::pipe::ITimelineDecoder::Relationship relationship28;
    relationship28.m_Guid = uint64_t(28);
    relationship28.m_RelationshipType = arm::pipe::ITimelineDecoder::RelationshipType::LabelLink;
    relationship28.m_HeadGuid = uint64_t(2);
    relationship28.m_TailGuid = uint64_t(18419179028513879730U);
    timelineDecoder.CreateRelationship(relationship28);

    arm::pipe::ITimelineDecoder::Relationship relationship29;
    relationship29.m_Guid = uint64_t(29);
    relationship29.m_RelationshipType = arm::pipe::ITimelineDecoder::RelationshipType::LabelLink;
    relationship29.m_HeadGuid = uint64_t(28);
    relationship29.m_TailGuid = uint64_t(10420554295983197538U);
    timelineDecoder.CreateRelationship(relationship29);

    arm::pipe::ITimelineDecoder::Relationship relationship30;
    relationship30.m_Guid = uint64_t(30);
    relationship30.m_RelationshipType = arm::pipe::ITimelineDecoder::RelationshipType::LabelLink;
    relationship30.m_HeadGuid = uint64_t(2);
    relationship30.m_TailGuid = uint64_t(14761340794127440397U);
    timelineDecoder.CreateRelationship(relationship30);

    arm::pipe::ITimelineDecoder::Relationship relationship31;
    relationship31.m_Guid = uint64_t(31);
    relationship31.m_RelationshipType = arm::pipe::ITimelineDecoder::RelationshipType::LabelLink;
    relationship31.m_HeadGuid = uint64_t(30);
    relationship31.m_TailGuid = uint64_t(14196220359693045352U);
    timelineDecoder.CreateRelationship(relationship31);

    arm::pipe::ITimelineDecoder::Relationship relationship32;
    relationship32.m_Guid = uint64_t(32);
    relationship32.m_RelationshipType = arm::pipe::ITimelineDecoder::RelationshipType::RetentionLink;
    relationship32.m_HeadGuid = uint64_t(6);
    relationship32.m_TailGuid = uint64_t(2);
    timelineDecoder.CreateRelationship(relationship32);

    arm::pipe::ITimelineDecoder::Relationship relationship33;
    relationship33.m_Guid = uint64_t(33);
    relationship33.m_RelationshipType = arm::pipe::ITimelineDecoder::RelationshipType::RetentionLink;
    relationship33.m_HeadGuid = uint64_t(1);
    relationship33.m_TailGuid = uint64_t(2);
    timelineDecoder.CreateRelationship(relationship33);

    arm::pipe::ITimelineDecoder::Relationship relationship34;
    relationship34.m_Guid = uint64_t(34);
    relationship34.m_RelationshipType = arm::pipe::ITimelineDecoder::RelationshipType::LabelLink;
    relationship34.m_HeadGuid = uint64_t(33);
    relationship34.m_TailGuid = uint64_t(15733717748792475675U);
    timelineDecoder.CreateRelationship(relationship34);

    arm::pipe::ITimelineDecoder::Relationship relationship35;
    relationship35.m_Guid = uint64_t(35);
    relationship35.m_RelationshipType = arm::pipe::ITimelineDecoder::RelationshipType::LabelLink;
    relationship35.m_HeadGuid = uint64_t(34);
    relationship35.m_TailGuid = uint64_t(14196220359693045352U);
    timelineDecoder.CreateRelationship(relationship35);


    arm::pipe::ITimelineDecoder::Entity entity36;
    entity36.m_Guid = uint64_t(36);
    timelineDecoder.CreateEntity(entity36);

    arm::pipe::ITimelineDecoder::Relationship relationship37;
    relationship37.m_Guid = uint64_t(37);
    relationship37.m_RelationshipType = arm::pipe::ITimelineDecoder::RelationshipType::LabelLink;
    relationship37.m_HeadGuid = uint64_t(36);
    relationship37.m_TailGuid = uint64_t(15704252740552608110U);
    timelineDecoder.CreateRelationship(relationship37);

    arm::pipe::ITimelineDecoder::Relationship relationship38;
    relationship38.m_Guid = uint64_t(38);
    relationship38.m_RelationshipType = arm::pipe::ITimelineDecoder::RelationshipType::LabelLink;
    relationship38.m_HeadGuid = uint64_t(37);
    relationship38.m_TailGuid = uint64_t(14196220359693045352U);
    timelineDecoder.CreateRelationship(relationship38);

    arm::pipe::ITimelineDecoder::Relationship relationship39;
    relationship39.m_Guid = uint64_t(39);
    relationship39.m_RelationshipType = arm::pipe::ITimelineDecoder::RelationshipType::LabelLink;
    relationship39.m_HeadGuid = uint64_t(36);
    relationship39.m_TailGuid = uint64_t(9690680943817437852U);
    timelineDecoder.CreateRelationship(relationship39);

    arm::pipe::ITimelineDecoder::Relationship relationship40;
    relationship40.m_Guid = uint64_t(40);
    relationship40.m_RelationshipType = arm::pipe::ITimelineDecoder::RelationshipType::LabelLink;
    relationship40.m_HeadGuid = uint64_t(39);
    relationship40.m_TailGuid = uint64_t(10874037804557439415U);
    timelineDecoder.CreateRelationship(relationship40);

    arm::pipe::ITimelineDecoder::Relationship relationship41;
    relationship41.m_Guid = uint64_t(41);
    relationship41.m_RelationshipType = arm::pipe::ITimelineDecoder::RelationshipType::RetentionLink;
    relationship41.m_HeadGuid = uint64_t(0);
    relationship41.m_TailGuid = uint64_t(36);
    timelineDecoder.CreateRelationship(relationship41);


    arm::pipe::ITimelineDecoder::Entity entity42;
    entity42.m_Guid = uint64_t(42);
    timelineDecoder.CreateEntity(entity42);

    arm::pipe::ITimelineDecoder::Relationship relationship43;
    relationship43.m_Guid = uint64_t(43);
    relationship43.m_RelationshipType = arm::pipe::ITimelineDecoder::RelationshipType::LabelLink;
    relationship43.m_HeadGuid = uint64_t(42);
    relationship43.m_TailGuid = uint64_t(15704252740552608110U);
    timelineDecoder.CreateRelationship(relationship43);

    arm::pipe::ITimelineDecoder::Relationship relationship44;
    relationship44.m_Guid = uint64_t(44);
    relationship44.m_RelationshipType = arm::pipe::ITimelineDecoder::RelationshipType::LabelLink;
    relationship44.m_HeadGuid = uint64_t(43);
    relationship44.m_TailGuid = uint64_t(14196220359693045352U);
    timelineDecoder.CreateRelationship(relationship44);

    arm::pipe::ITimelineDecoder::Relationship relationship45;
    relationship45.m_Guid = uint64_t(45);
    relationship45.m_RelationshipType = arm::pipe::ITimelineDecoder::RelationshipType::LabelLink;
    relationship45.m_HeadGuid = uint64_t(42);
    relationship45.m_TailGuid = uint64_t(9690680943817437852U);
    timelineDecoder.CreateRelationship(relationship45);

    arm::pipe::ITimelineDecoder::Relationship relationship46;
    relationship46.m_Guid = uint64_t(46);
    relationship46.m_RelationshipType = arm::pipe::ITimelineDecoder::RelationshipType::LabelLink;
    relationship46.m_HeadGuid = uint64_t(45);
    relationship46.m_TailGuid = uint64_t(10874037804557439415U);
    timelineDecoder.CreateRelationship(relationship46);

    arm::pipe::ITimelineDecoder::Relationship relationship47;
    relationship47.m_Guid = uint64_t(47);
    relationship47.m_RelationshipType = arm::pipe::ITimelineDecoder::RelationshipType::RetentionLink;
    relationship47.m_HeadGuid = uint64_t(2);
    relationship47.m_TailGuid = uint64_t(42);
    timelineDecoder.CreateRelationship(relationship47);

    arm::pipe::ITimelineDecoder::Entity entity48;
    entity48.m_Guid = uint64_t(48);
    timelineDecoder.CreateEntity(entity48);

    arm::pipe::ITimelineDecoder::Relationship relationship49;
    relationship49.m_Guid = uint64_t(49);
    relationship49.m_RelationshipType = arm::pipe::ITimelineDecoder::RelationshipType::LabelLink;
    relationship49.m_HeadGuid = uint64_t(48);
    relationship49.m_TailGuid = uint64_t(15026600058430441282U);
    timelineDecoder.CreateRelationship(relationship49);

    arm::pipe::ITimelineDecoder::Relationship relationship50;
    relationship50.m_Guid = uint64_t(50);
    relationship50.m_RelationshipType = arm::pipe::ITimelineDecoder::RelationshipType::LabelLink;
    relationship50.m_HeadGuid = uint64_t(49);
    relationship50.m_TailGuid = uint64_t(14196220359693045352U);
    timelineDecoder.CreateRelationship(relationship50);

    arm::pipe::ITimelineDecoder::Relationship relationship51;
    relationship51.m_Guid = uint64_t(51);
    relationship51.m_RelationshipType = arm::pipe::ITimelineDecoder::RelationshipType::RetentionLink;
    relationship51.m_HeadGuid = uint64_t(6);
    relationship51.m_TailGuid = uint64_t(48);
    timelineDecoder.CreateRelationship(relationship51);

    arm::pipe::ITimelineDecoder::Relationship relationship53;
    relationship53.m_Guid = uint64_t(53);
    relationship53.m_RelationshipType = arm::pipe::ITimelineDecoder::RelationshipType::DataLink;
    relationship53.m_HeadGuid = uint64_t(48);
    relationship53.m_TailGuid = uint64_t(52);
    timelineDecoder.CreateRelationship(relationship53);

    arm::pipe::ITimelineDecoder::Relationship relationship54;
    relationship54.m_Guid = uint64_t(54);
    relationship54.m_RelationshipType = arm::pipe::ITimelineDecoder::RelationshipType::ExecutionLink;
    relationship54.m_HeadGuid = uint64_t(52);
    relationship54.m_TailGuid = uint64_t(17170418158534996719U);
    timelineDecoder.CreateRelationship(relationship54);


    arm::pipe::ITimelineDecoder::Entity entity55;
    entity55.m_Guid = uint64_t(55);
    timelineDecoder.CreateEntity(entity55);

    arm::pipe::ITimelineDecoder::Relationship relationship56;
    relationship56.m_Guid = uint64_t(56);
    relationship56.m_RelationshipType = arm::pipe::ITimelineDecoder::RelationshipType::LabelLink;
    relationship56.m_HeadGuid = uint64_t(55);
    relationship56.m_TailGuid = uint64_t(10172155312650606003U);
    timelineDecoder.CreateRelationship(relationship56);

    arm::pipe::ITimelineDecoder::Relationship relationship57;
    relationship57.m_Guid = uint64_t(57);
    relationship57.m_RelationshipType = arm::pipe::ITimelineDecoder::RelationshipType::LabelLink;
    relationship57.m_HeadGuid = uint64_t(56);
    relationship57.m_TailGuid = uint64_t(14196220359693045352U);
    timelineDecoder.CreateRelationship(relationship57);

    arm::pipe::ITimelineDecoder::Relationship relationship58;
    relationship58.m_Guid = uint64_t(58);
    relationship58.m_RelationshipType = arm::pipe::ITimelineDecoder::RelationshipType::RetentionLink;
    relationship58.m_HeadGuid = uint64_t(48);
    relationship58.m_TailGuid = uint64_t(55);
    timelineDecoder.CreateRelationship(relationship58);

    arm::pipe::ITimelineDecoder::Relationship relationship59;
    relationship59.m_Guid = uint64_t(59);
    relationship59.m_RelationshipType = arm::pipe::ITimelineDecoder::RelationshipType::RetentionLink;
    relationship59.m_HeadGuid = uint64_t(36);
    relationship59.m_TailGuid = uint64_t(55);
    timelineDecoder.CreateRelationship(relationship59);

    arm::pipe::ITimelineDecoder::Event event60;
    event60.m_Guid = uint64_t(60);
    event60.m_TimeStamp = uint64_t(96557081111036);
    event60.m_ThreadId = uint64_t(140522431862592);
    timelineDecoder.CreateEvent(event60);

    arm::pipe::ITimelineDecoder::Relationship relationship61;
    relationship61.m_Guid = uint64_t(61);
    relationship61.m_RelationshipType = arm::pipe::ITimelineDecoder::RelationshipType::ExecutionLink;
    relationship61.m_HeadGuid = uint64_t(55);
    relationship61.m_TailGuid = uint64_t(60);
    timelineDecoder.CreateRelationship(relationship61);

    arm::pipe::ITimelineDecoder::Relationship relationship62;
    relationship62.m_Guid = uint64_t(62);
    relationship62.m_RelationshipType = arm::pipe::ITimelineDecoder::RelationshipType::DataLink;
    relationship62.m_HeadGuid = uint64_t(60);
    relationship62.m_TailGuid = uint64_t(17170418158534996719U);
    timelineDecoder.CreateRelationship(relationship62);

    arm::pipe::ITimelineDecoder::Event event63;
    event63.m_Guid = uint64_t(63);
    event63.m_TimeStamp = uint64_t(96557081149730);
    event63.m_ThreadId = uint64_t(140522431862592);
    timelineDecoder.CreateEvent(event63);

    arm::pipe::ITimelineDecoder::Relationship relationship64;
    relationship64.m_Guid = uint64_t(61);
    relationship64.m_RelationshipType = arm::pipe::ITimelineDecoder::RelationshipType::ExecutionLink;
    relationship64.m_HeadGuid = uint64_t(55);
    relationship64.m_TailGuid = uint64_t(63);
    timelineDecoder.CreateRelationship(relationship64);

    arm::pipe::ITimelineDecoder::Relationship relationship65;
    relationship65.m_Guid = uint64_t(62);
    relationship65.m_RelationshipType = arm::pipe::ITimelineDecoder::RelationshipType::DataLink;
    relationship65.m_HeadGuid = uint64_t(63);
    relationship65.m_TailGuid = uint64_t(10812061579584851344U);
    timelineDecoder.CreateRelationship(relationship65);


    arm::pipe::ITimelineDecoder::Entity entity66;
    entity66.m_Guid = uint64_t(66);
    timelineDecoder.CreateEntity(entity66);

    arm::pipe::ITimelineDecoder::Relationship relationship67;
    relationship67.m_Guid = uint64_t(67);
    relationship67.m_RelationshipType = arm::pipe::ITimelineDecoder::RelationshipType::LabelLink;
    relationship67.m_HeadGuid = uint64_t(66);
    relationship67.m_TailGuid = uint64_t(10172155312650606003U);
    timelineDecoder.CreateRelationship(relationship67);

    arm::pipe::ITimelineDecoder::Relationship relationship68;
    relationship68.m_Guid = uint64_t(68);
    relationship68.m_RelationshipType = arm::pipe::ITimelineDecoder::RelationshipType::LabelLink;
    relationship68.m_HeadGuid = uint64_t(67);
    relationship68.m_TailGuid = uint64_t(14196220359693045352U);
    timelineDecoder.CreateRelationship(relationship68);

    arm::pipe::ITimelineDecoder::Relationship relationship69;
    relationship69.m_Guid = uint64_t(69);
    relationship69.m_RelationshipType = arm::pipe::ITimelineDecoder::RelationshipType::RetentionLink;
    relationship69.m_HeadGuid = uint64_t(48);
    relationship69.m_TailGuid = uint64_t(66);
    timelineDecoder.CreateRelationship(relationship69);

    arm::pipe::ITimelineDecoder::Relationship relationship70;
    relationship70.m_Guid = uint64_t(70);
    relationship70.m_RelationshipType = arm::pipe::ITimelineDecoder::RelationshipType::RetentionLink;
    relationship70.m_HeadGuid = uint64_t(22);
    relationship70.m_TailGuid = uint64_t(66);
    timelineDecoder.CreateRelationship(relationship70);


    arm::pipe::ITimelineDecoder::Event event71;
    event71.m_Guid = uint64_t(71);
    event71.m_TimeStamp = uint64_t(96557081156464);
    event71.m_ThreadId = uint64_t(140522431862592);
    timelineDecoder.CreateEvent(event71);

    arm::pipe::ITimelineDecoder::Relationship relationship72;
    relationship72.m_Guid = uint64_t(72);
    relationship72.m_RelationshipType = arm::pipe::ITimelineDecoder::RelationshipType::ExecutionLink;
    relationship72.m_HeadGuid = uint64_t(66);
    relationship72.m_TailGuid = uint64_t(71);
    timelineDecoder.CreateRelationship(relationship72);

    arm::pipe::ITimelineDecoder::Relationship relationship73;
    relationship73.m_Guid = uint64_t(73);
    relationship73.m_RelationshipType = arm::pipe::ITimelineDecoder::RelationshipType::DataLink;
    relationship73.m_HeadGuid = uint64_t(71);
    relationship73.m_TailGuid = uint64_t(17170418158534996719U);
    timelineDecoder.CreateRelationship(relationship73);

    arm::pipe::ITimelineDecoder::Event event74;
    event74.m_Guid = uint64_t(74);
    event74.m_TimeStamp = uint64_t(96557081220825);
    event74.m_ThreadId = uint64_t(140522431862592);
    timelineDecoder.CreateEvent(event74);

    arm::pipe::ITimelineDecoder::Relationship relationship75;
    relationship75.m_Guid = uint64_t(75);
    relationship75.m_RelationshipType = arm::pipe::ITimelineDecoder::RelationshipType::ExecutionLink;
    relationship75.m_HeadGuid = uint64_t(66);
    relationship75.m_TailGuid = uint64_t(74);
    timelineDecoder.CreateRelationship(relationship75);

    arm::pipe::ITimelineDecoder::Relationship relationship76;
    relationship76.m_Guid = uint64_t(76);
    relationship76.m_RelationshipType = arm::pipe::ITimelineDecoder::RelationshipType::DataLink;
    relationship76.m_HeadGuid = uint64_t(74);
    relationship76.m_TailGuid = uint64_t(10812061579584851344U);
    timelineDecoder.CreateRelationship(relationship76);

    arm::pipe::ITimelineDecoder::Entity entity77;
    entity77.m_Guid = uint64_t(77);
    timelineDecoder.CreateEntity(entity77);

    arm::pipe::ITimelineDecoder::Relationship relationship78;
    relationship78.m_Guid = uint64_t(78);
    relationship78.m_RelationshipType = arm::pipe::ITimelineDecoder::RelationshipType::LabelLink;
    relationship78.m_HeadGuid = uint64_t(77);
    relationship78.m_TailGuid = uint64_t(10172155312650606003U);
    timelineDecoder.CreateRelationship(relationship78);

    arm::pipe::ITimelineDecoder::Relationship relationship79;
    relationship79.m_Guid = uint64_t(79);
    relationship79.m_RelationshipType = arm::pipe::ITimelineDecoder::RelationshipType::LabelLink;
    relationship79.m_HeadGuid = uint64_t(78);
    relationship79.m_TailGuid = uint64_t(14196220359693045352U);
    timelineDecoder.CreateRelationship(relationship79);

    arm::pipe::ITimelineDecoder::Relationship relationship80;
    relationship80.m_Guid = uint64_t(80);
    relationship80.m_RelationshipType = arm::pipe::ITimelineDecoder::RelationshipType::RetentionLink;
    relationship80.m_HeadGuid = uint64_t(48);
    relationship80.m_TailGuid = uint64_t(77);
    timelineDecoder.CreateRelationship(relationship80);

    arm::pipe::ITimelineDecoder::Relationship relationship81;
    relationship81.m_Guid = uint64_t(81);
    relationship81.m_RelationshipType = arm::pipe::ITimelineDecoder::RelationshipType::RetentionLink;
    relationship81.m_HeadGuid = uint64_t(42);
    relationship81.m_TailGuid = uint64_t(77);
    timelineDecoder.CreateRelationship(relationship81);

    arm::pipe::ITimelineDecoder::Event event82;
    event82.m_Guid = uint64_t(82);
    event82.m_TimeStamp = uint64_t(96557081227643);
    event82.m_ThreadId = uint64_t(140522431862592);
    timelineDecoder.CreateEvent(event82);

    arm::pipe::ITimelineDecoder::Relationship relationship83;
    relationship83.m_Guid = uint64_t(83);
    relationship83.m_RelationshipType = arm::pipe::ITimelineDecoder::RelationshipType::ExecutionLink;
    relationship83.m_HeadGuid = uint64_t(77);
    relationship83.m_TailGuid = uint64_t(82);
    timelineDecoder.CreateRelationship(relationship83);

    arm::pipe::ITimelineDecoder::Relationship relationship84;
    relationship84.m_Guid = uint64_t(84);
    relationship84.m_RelationshipType = arm::pipe::ITimelineDecoder::RelationshipType::DataLink;
    relationship84.m_HeadGuid = uint64_t(82);
    relationship84.m_TailGuid = uint64_t(17170418158534996719U);
    timelineDecoder.CreateRelationship(relationship84);

    arm::pipe::ITimelineDecoder::Event event85;
    event85.m_Guid = uint64_t(85);
    event85.m_TimeStamp = uint64_t(96557081240332);
    event85.m_ThreadId = uint64_t(140522431862592);
    timelineDecoder.CreateEvent(event85);

    arm::pipe::ITimelineDecoder::Relationship relationship86;
    relationship86.m_Guid = uint64_t(86);
    relationship86.m_RelationshipType = arm::pipe::ITimelineDecoder::RelationshipType::ExecutionLink;
    relationship86.m_HeadGuid = uint64_t(77);
    relationship86.m_TailGuid = uint64_t(85);
    timelineDecoder.CreateRelationship(relationship86);

    arm::pipe::ITimelineDecoder::Relationship relationship87;
    relationship87.m_Guid = uint64_t(87);
    relationship87.m_RelationshipType = arm::pipe::ITimelineDecoder::RelationshipType::DataLink;
    relationship87.m_HeadGuid = uint64_t(85);
    relationship87.m_TailGuid = uint64_t(10812061579584851344U);
    timelineDecoder.CreateRelationship(relationship87);

    arm::pipe::ITimelineDecoder::Event event88;
    event88.m_Guid = uint64_t(88);
    event88.m_TimeStamp = uint64_t(96557081243146);
    event88.m_ThreadId = uint64_t(140522431862592);
    timelineDecoder.CreateEvent(event88);

    arm::pipe::ITimelineDecoder::Relationship relationship89;
    relationship89.m_Guid = uint64_t(89);
    relationship89.m_RelationshipType = arm::pipe::ITimelineDecoder::RelationshipType::ExecutionLink;
    relationship89.m_HeadGuid = uint64_t(48);
    relationship89.m_TailGuid = uint64_t(88);
    timelineDecoder.CreateRelationship(relationship89);

    arm::pipe::ITimelineDecoder::Relationship relationship90;
    relationship90.m_Guid = uint64_t(90);
    relationship90.m_RelationshipType = arm::pipe::ITimelineDecoder::RelationshipType::DataLink;
    relationship90.m_HeadGuid = uint64_t(88);
    relationship90.m_TailGuid = uint64_t(10812061579584851344U);
    timelineDecoder.CreateRelationship(relationship90);
}

TEST_CASE("JSONTimelineDecoderTestStructure")
{
    JSONTimelineDecoder timelineDecoder;
    RunSimpleModelThroughDecoder(timelineDecoder);

    JSONTimelineDecoder::Model model = timelineDecoder.GetModel();
    CHECK(model.jsonEntities.size() == 20);
    JSONTimelineDecoder::JSONEntity rootEntity = model.jsonEntities.at(6);
    CHECK(rootEntity.childEntities.size() == 4);

    // Testing input layer model
    JSONTimelineDecoder::JSONEntity entity0 = model.jsonEntities.at(rootEntity.childEntities[0]);
    CHECK(entity0.GetName() == "input");
    CHECK(entity0.GetType() == "layer");

    CHECK(entity0.childEntities.size() == 1);
    JSONTimelineDecoder::JSONEntity input_workload_entity = model.jsonEntities.at(entity0.childEntities[0]);
    CHECK(input_workload_entity.childEntities.size() == 1);
    CHECK(input_workload_entity.GetType() == "workload");
    CHECK(input_workload_entity.extendedData.at("backendId") == "CpuRef");

    JSONTimelineDecoder::JSONEntity input_workload_execution_entity = model.jsonEntities
            .at(input_workload_entity.childEntities[0]);
    CHECK(input_workload_execution_entity.childEntities.size() == 2);
    CHECK(input_workload_execution_entity.GetType() == "workload_execution");

    JSONTimelineDecoder::JSONEntity input_workload_execution_event0 = model.jsonEntities
            .at(input_workload_execution_entity.childEntities[0]);
    CHECK(input_workload_execution_event0.GetType() == "Event");
    CHECK(input_workload_execution_event0.childEntities.size() == 0);
    CHECK(model.events.at(input_workload_execution_event0.GetGuid()).m_ThreadId > uint64_t(0));
    CHECK(model.events.at(input_workload_execution_event0.GetGuid()).m_TimeStamp > uint64_t(0));

    JSONTimelineDecoder::JSONEntity input_workload_execution_event1 = model.jsonEntities
            .at(input_workload_execution_entity.childEntities[1]);
    CHECK(input_workload_execution_event0.GetType() == "Event");
    CHECK(input_workload_execution_event1.childEntities.size() == 0);
    CHECK(model.events.at(input_workload_execution_event1.GetGuid()).m_ThreadId > uint64_t(0));
    CHECK(model.events.at(input_workload_execution_event1.GetGuid()).m_TimeStamp > uint64_t(0));

    // Testing normalization layer model
    JSONTimelineDecoder::JSONEntity entity1 = model.jsonEntities.at(rootEntity.childEntities[1]);
    CHECK(entity1.GetName() == "normalization");
    CHECK(entity1.GetType() == "layer");

    JSONTimelineDecoder::JSONEntity normalization_workload_entity = model.jsonEntities
            .at(entity1.childEntities[0]);
    CHECK(normalization_workload_entity.GetType() == "workload");
    CHECK(normalization_workload_entity.extendedData.at("backendId") == "CpuRef");

    JSONTimelineDecoder::JSONEntity normalization_workload_execution_entity = model.jsonEntities
            .at(normalization_workload_entity.childEntities[0]);
    CHECK(normalization_workload_execution_entity.GetType() == "workload_execution");

    JSONTimelineDecoder::JSONEntity normalization_workload_execution_event0 = model.jsonEntities
            .at(normalization_workload_execution_entity.childEntities[0]);
    CHECK(normalization_workload_execution_event0.GetType() == "Event");
    CHECK(model.events.at(normalization_workload_execution_event0.GetGuid()).m_ThreadId > uint64_t(0));
    CHECK(model.events.at(normalization_workload_execution_event0.GetGuid()).m_TimeStamp > uint64_t(0));

    JSONTimelineDecoder::JSONEntity normalization_workload_execution_event1 = model.jsonEntities
            .at(normalization_workload_execution_entity.childEntities[1]);
    CHECK(normalization_workload_execution_event1.GetType() == "Event");
    CHECK(model.events.at(normalization_workload_execution_event1.GetGuid()).m_ThreadId > uint64_t(0));
    CHECK(model.events.at(normalization_workload_execution_event1.GetGuid()).m_TimeStamp > uint64_t(0));

    // Testing output layer model
    JSONTimelineDecoder::JSONEntity entity2 = model.jsonEntities.at(rootEntity.childEntities[2]);
    CHECK(entity2.GetName() == "output");
    CHECK(entity2.GetType() == "layer");

    JSONTimelineDecoder::JSONEntity output_workload_entity = model.jsonEntities.at(entity2.childEntities[0]);
    CHECK(output_workload_entity.GetType() == "workload");
    CHECK(output_workload_entity.extendedData.at("backendId") == "CpuRef");

    JSONTimelineDecoder::JSONEntity output_workload_execution_entity = model.jsonEntities
            .at(output_workload_entity.childEntities[0]);
    CHECK(output_workload_execution_entity.GetType() == "workload_execution");

    JSONTimelineDecoder::JSONEntity output_workload_execution_event0 = model.jsonEntities
            .at(output_workload_execution_entity.childEntities[0]);
    CHECK(output_workload_execution_event0.GetType() == "Event");
    CHECK(model.events.at(output_workload_execution_event0.GetGuid()).m_ThreadId > uint64_t(0));
    CHECK(model.events.at(output_workload_execution_event0.GetGuid()).m_TimeStamp > uint64_t(0));

    JSONTimelineDecoder::JSONEntity output_workload_execution_event1 = model.jsonEntities
            .at(output_workload_execution_entity.childEntities[1]);
    CHECK(output_workload_execution_event1.GetType() == "Event");
    CHECK(model.events.at(output_workload_execution_event1.GetGuid()).m_ThreadId > uint64_t(0));
    CHECK(model.events.at(output_workload_execution_event1.GetGuid()).m_TimeStamp > uint64_t(0));

    JSONTimelineDecoder::JSONEntity entity48 =  model.jsonEntities.at(rootEntity.childEntities[3]);
    CHECK(entity48.GetName() == "");
    CHECK(entity48.GetType() == "inference");
}

TEST_CASE("JSONTimelineDecoderTestJSON")
{
    JSONTimelineDecoder timelineDecoder;
    RunSimpleModelThroughDecoder(timelineDecoder);

    JSONTimelineDecoder::Model model = timelineDecoder.GetModel();
    JSONTimelineDecoder::JSONEntity rootEntity = model.jsonEntities.at(6);

    std::string jsonString = timelineDecoder.GetJSONString(rootEntity);
    CHECK(jsonString != "");
    CHECK(jsonString.find("input_0: {")!=std::string::npos);
    CHECK(jsonString.find("type: Measurement,\n"
                                   "\t\t\tbackendId :CpuRef,")!=std::string::npos);
    CHECK(jsonString.find("normalization_2: {")!=std::string::npos);
    CHECK(jsonString.find("output_4: {")!=std::string::npos);

    // Create a temporary file to write Json output to
    fs::path tempFile = armnnUtils::Filesystem::NamedTempFile("JSONTimelineDecoderTestJSON.json");
    // open temp file
    std::ofstream ofs{tempFile};
    // tell the timeline decoder to print into our temp file (you could also use std::cout)
    timelineDecoder.PrintJSON(rootEntity, ofs);
    // close temp file
    ofs.close();

    // Now everything in opposite order
    fs::ifstream inFile;
    //reopen the file this time for reading
    inFile.open(tempFile);

    std::stringstream strStream;
    strStream << inFile.rdbuf(); //read the file
    inFile.close();
    std::string outfileJson = strStream.str();

    CHECK(outfileJson != "");
    CHECK(outfileJson.find("input_0: {")!=std::string::npos);
    CHECK(outfileJson.find("type: Measurement,\n"
                                "\t\t\tbackendId :CpuRef,")!=std::string::npos);
    CHECK(outfileJson.find("normalization_2: {")!=std::string::npos);
    CHECK(outfileJson.find("output_4: {")!=std::string::npos);

    // Remove temporary file
    fs::remove(tempFile);
}
}