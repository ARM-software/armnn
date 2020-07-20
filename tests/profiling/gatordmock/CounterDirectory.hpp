//
// Copyright © 2019 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once


#include "GatordMockService.hpp"
#include "MockUtils.hpp"

#include <common/include/Packet.hpp>
#include <common/include/CommandHandlerFunctor.hpp>

#include "SendCounterPacket.hpp"
#include "IPeriodicCounterCapture.hpp"

#include <vector>
#include <thread>
#include <atomic>
#include <iostream>
#include <functional>

namespace armnn
{

namespace gatordmock
{

struct EventRecord
{
    uint16_t m_CounterUid;
    uint16_t m_MaxCounterUid;
    uint16_t m_DeviceUid;
    uint16_t m_CounterSetUid;
    uint16_t m_CounterClass;
    uint16_t m_CounterInterpolation;
    double m_CounterMultiplier;
    std::string m_CounterName;
    std::string m_CounterDescription;
    std::string m_CounterUnits;

    static void printHeader(std::string categoryName)
    {
        std::string header;

        header.append(gatordmock::CentreAlignFormatting("Counter Name", 20));
        header.append(" | ");
        header.append(gatordmock::CentreAlignFormatting("Description", 50));
        header.append(" | ");
        header.append(gatordmock::CentreAlignFormatting("Units", 14));
        header.append(" | ");
        header.append(gatordmock::CentreAlignFormatting("UID", 6));
        header.append(" | ");
        header.append(gatordmock::CentreAlignFormatting("Max UID",10));
        header.append(" | ");
        header.append(gatordmock::CentreAlignFormatting("Class", 8));
        header.append(" | ");
        header.append(gatordmock::CentreAlignFormatting("Interpolation", 14));
        header.append(" | ");
        header.append(gatordmock::CentreAlignFormatting("Multiplier", 20));
        header.append(" | ");
        header.append(gatordmock::CentreAlignFormatting("Counter set UID", 16));
        header.append(" | ");
        header.append(gatordmock::CentreAlignFormatting("Device UID", 14));
        header.append("\n");

        std::cout << "\n" << "\n";
        std::cout << gatordmock::CentreAlignFormatting("EVENTS IN CATEGORY: " + categoryName,
                                                       static_cast<int>(header.size()));
        std::cout << "\n";
        std::cout << std::string(header.size(), '=') << "\n";
        std::cout << header;
    }

    void printContents() const
    {
        std::string body;

        body.append(gatordmock::CentreAlignFormatting(m_CounterName, 20));
        body.append(" | ");
        body.append(gatordmock::CentreAlignFormatting(m_CounterDescription, 50));
        body.append(" | ");
        body.append(gatordmock::CentreAlignFormatting(m_CounterUnits, 14));
        body.append(" | ");
        body.append(gatordmock::CentreAlignFormatting(std::to_string(m_CounterUid), 6));
        body.append(" | ");
        body.append(gatordmock::CentreAlignFormatting(std::to_string(m_MaxCounterUid), 10));
        body.append(" | ");
        body.append(gatordmock::CentreAlignFormatting(std::to_string(m_CounterClass), 8));
        body.append(" | ");
        body.append(gatordmock::CentreAlignFormatting(std::to_string(m_CounterInterpolation), 14));
        body.append(" | ");
        body.append(gatordmock::CentreAlignFormatting(std::to_string(m_CounterMultiplier), 20));
        body.append(" | ");
        body.append(gatordmock::CentreAlignFormatting(std::to_string(m_CounterSetUid), 16));
        body.append(" | ");
        body.append(gatordmock::CentreAlignFormatting(std::to_string(m_DeviceUid), 14));

        body.append("\n");

        std::cout << std::string(body.size(), '-') << "\n";

        std::cout << body;
    }
};

struct CategoryRecord
{
    uint16_t m_EventCount;
    std::string m_CategoryName;
    std::vector<EventRecord> m_EventRecords;

    void print() const
    {
        std::string body;
        std::string header;

        header.append(gatordmock::CentreAlignFormatting("Name", 20));
        header.append(" | ");
        header.append(gatordmock::CentreAlignFormatting("Event Count", 14));
        header.append("\n");

        body.append(gatordmock::CentreAlignFormatting(m_CategoryName, 20));
        body.append(" | ");
        body.append(gatordmock::CentreAlignFormatting(std::to_string(m_EventCount), 14));

        std::cout << "\n" << "\n";
        std::cout << gatordmock::CentreAlignFormatting("CATEGORY", static_cast<int>(header.size()));
        std::cout << "\n";
        std::cout << std::string(header.size(), '=') << "\n";

        std::cout<< header;

        std::cout << std::string(body.size(), '-') << "\n";

        std::cout<< body;

        if(m_EventRecords.size() > 0)
        {
            EventRecord::printHeader(m_CategoryName);

            std::for_each(m_EventRecords.begin(), m_EventRecords.end(), std::mem_fun_ref(&EventRecord::printContents));
        }
    }
};

struct CounterSetRecord
{
    uint16_t m_CounterSetUid;
    uint16_t m_CounterSetCount;
    std::string m_CounterSetName;

    static void printHeader()
    {
        std::string header;

        header.append(gatordmock::CentreAlignFormatting("Counter set name", 20));
        header.append(" | ");
        header.append(gatordmock::CentreAlignFormatting("UID",13));
        header.append(" | ");
        header.append(gatordmock::CentreAlignFormatting("Count",10));
        header.append("\n");

        std::cout << "\n" << "\n";
        std::cout << gatordmock::CentreAlignFormatting("COUNTER SETS", static_cast<int>(header.size()));
        std::cout << "\n";
        std::cout << std::string(header.size(), '=') << "\n";

        std::cout<< header;
    }

    void printContents() const
    {
        std::string body;

        body.append(gatordmock::CentreAlignFormatting(m_CounterSetName, 20));
        body.append(" | ");
        body.append(gatordmock::CentreAlignFormatting(std::to_string(m_CounterSetUid), 13));
        body.append(" | ");
        body.append(gatordmock::CentreAlignFormatting(std::to_string(m_CounterSetCount), 10));
        body.append("\n");

        std::cout << std::string(body.size(), '-') << "\n";

        std::cout<< body;
    }
};

struct DeviceRecord
{
    uint16_t m_DeviceUid;
    uint16_t m_DeviceCores;
    std::string m_DeviceName;

    static void printHeader()
    {
        std::string header;

        header.append(gatordmock::CentreAlignFormatting("Device name", 20));
        header.append(" | ");
        header.append(gatordmock::CentreAlignFormatting("UID",13));
        header.append(" | ");
        header.append(gatordmock::CentreAlignFormatting("Cores",10));
        header.append("\n");

        std::cout << "\n" << "\n";
        std::cout << gatordmock::CentreAlignFormatting("DEVICES", static_cast<int>(header.size()));
        std::cout << "\n";
        std::cout << std::string(header.size(), '=') << "\n";
        std::cout<< header;
    }

    void printContents() const
    {
        std::string body;

        body.append(gatordmock::CentreAlignFormatting(m_DeviceName, 20));
        body.append(" | ");
        body.append(gatordmock::CentreAlignFormatting(std::to_string(m_DeviceUid), 13));
        body.append(" | ");
        body.append(gatordmock::CentreAlignFormatting(std::to_string(m_DeviceCores), 10));
        body.append("\n");

        std::cout << std::string(body.size(), '-') << "\n";
        std::cout<< body;
    }
};

struct CounterDirectory
{
    std::vector<CategoryRecord> m_Categories;
    std::vector<CounterSetRecord> m_CounterSets;
    std::vector<DeviceRecord> m_DeviceRecords;

    void print() const
    {
        DeviceRecord::printHeader();
        std::for_each(m_DeviceRecords.begin(), m_DeviceRecords.end(),
                      std::mem_fun_ref(&DeviceRecord::printContents));

        CounterSetRecord::printHeader();
        std::for_each(m_CounterSets.begin(), m_CounterSets.end(),
                      std::mem_fun_ref(&CounterSetRecord::printContents));

        std::for_each(m_Categories.begin(), m_Categories.end(),
                      std::mem_fun_ref(&CategoryRecord::print));
        std::cout << "\n";
    }
};

} // namespace gatordmock

} // namespace armnn