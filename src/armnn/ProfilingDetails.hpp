//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <iomanip>

#include "armnn/Types.hpp"
#include "armnn/TypesUtils.hpp"
#include "armnn/backends/WorkloadInfo.hpp"

#include "SerializeLayerParameters.hpp"
#include "JsonUtils.hpp"

namespace armnn
{

/// ProfilingDetails class records any details associated with the operator and passes on for outputting to the user
class ProfilingDetails : public JsonUtils
{
public:
    /// Constructor
    ProfilingDetails() : JsonUtils(m_ProfilingDetails), m_DetailsExist(false), m_PrintSeparator(false)
    {}

    /// Destructor
    ~ProfilingDetails() noexcept
    {}

    /// Add to the ProfilingDetails
    template<typename DescriptorType>
    void AddDetailsToString(const std::string& workloadName,
                            const DescriptorType& desc,
                            const WorkloadInfo& infos,
                            const profiling::ProfilingGuid guid)
    {
        // Once details exist, we can assume we're on the second iteration of details
        if (m_DetailsExist)
        {
            PrintSeparator();
            PrintNewLine();
        }

        PrintHeader();
        PrintTabs();
        m_ProfilingDetails << std::quoted("Name") << ": " << std::quoted(workloadName);
        PrintSeparator();
        PrintNewLine();
        PrintTabs();
        m_ProfilingDetails << std::quoted("GUID") << ": " << std::quoted(std::to_string(guid));
        PrintSeparator();
        PrintNewLine();

        // Print tensor infos and related data types
        PrintInfos(infos.m_InputTensorInfos, "Input");

        PrintInfos(infos.m_OutputTensorInfos, "Output");

        if ( infos.m_BiasTensorInfo.has_value())
        {
            PrintInfo(infos.m_BiasTensorInfo.value(), "Bias");
        }
        if ( infos.m_WeightsTensorInfo.has_value())
        {
            PrintInfo(infos.m_WeightsTensorInfo.value(), "Weights");
        }
        if ( infos.m_ConvolutionMethod.has_value())
        {
            PrintTabs();

            m_ProfilingDetails << std::quoted("Convolution Method") << ": "
                               << std::quoted(infos.m_ConvolutionMethod.value());

            PrintSeparator();
            PrintNewLine();
        }

        ParameterStringifyFunction extractParams = [this](const std::string& name, const std::string& value) {
            if (m_PrintSeparator)
            {
                PrintSeparator();
                PrintNewLine();
            }
            PrintTabs();
            m_ProfilingDetails << std::quoted(name) << " : " << std::quoted(value);
            m_PrintSeparator = true;
        };

        StringifyLayerParameters<DescriptorType>::Serialize(extractParams, desc);

        PrintNewLine();
        PrintFooter();

        m_DetailsExist = true;
        m_PrintSeparator = false;
    }

    /// Get the ProfilingDetails
    /// \return the ProfilingDetails
    std::string GetProfilingDetails() const
    {
        return m_ProfilingDetails.str();
    }

    bool DetailsExist()
    {
        return m_DetailsExist;
    }

private:
    // Print tensor infos and related data types
    void PrintInfo(const TensorInfo& info, const std::string& ioString)
    {
        const std::vector<TensorInfo> infoVect{ info };
        PrintInfos(infoVect, ioString);
    }

    void PrintInfos(const std::vector<TensorInfo>& infos, const std::string& ioString)
    {
        for ( size_t i = 0; i < infos.size(); i++ )
        {
            auto shape = infos[i].GetShape();
            PrintTabs();

            m_ProfilingDetails << std::quoted(ioString + " " + std::to_string(i)) << ": ";

            PrintHeader();
            PrintTabs();

            // Shape
            m_ProfilingDetails << std::quoted("Shape") << ": \"[";
            for ( unsigned int dim = 0; dim < shape.GetNumDimensions(); dim++ )
            {
                shape.GetNumDimensions() == dim + 1 ?
                m_ProfilingDetails << shape[dim] << "]\"" : // true
                m_ProfilingDetails << shape[dim] << ",";    // false
            }

            PrintSeparator();
            PrintNewLine();

            // Data Type
            PrintTabs();
            m_ProfilingDetails << std::quoted("DataType") << ": "
                               << std::quoted(GetDataTypeName(infos[i].GetDataType()));

            PrintSeparator();
            PrintNewLine();

            // Number of Dimensions
            PrintTabs();
            m_ProfilingDetails << std::quoted("Num Dims") << ": "
                               << std::quoted(std::to_string(shape.GetNumDimensions()));


            // Close out the scope
            PrintNewLine();
            PrintFooter();
            PrintSeparator();
            PrintNewLine();
        }
    }

    /// Stores ProfilingDetails
    std::ostringstream m_ProfilingDetails;
    bool m_DetailsExist;
    bool m_PrintSeparator;

};

} // namespace armnn
