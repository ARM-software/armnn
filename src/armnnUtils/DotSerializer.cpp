//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "DotSerializer.hpp"
#include "armnn/utility/StringUtils.hpp"
#include <common/include/ProfilingGuid.hpp>

#include <sstream>
#include <cstring>

namespace armnn
{

namespace
{
std::string Indent(int numSpaces)
{
    std::stringstream ss;
    for (int i = 0; i < numSpaces; i++)
    {
        ss << " ";
    }
    return ss.str();
}

std::string Escape(std::string s)
{
    armnn::stringUtils::StringReplaceAll(s, "<", "\\<");
    armnn::stringUtils::StringReplaceAll(s, ">", "\\>");
    return s;
}

} //namespace


HtmlFont::HtmlFont(std::ostream& stream, int fontSize, const char *color, const char *face)
    : DotBase(stream)
{
    GetStream() << "<FONT";

    if (fontSize > -1)
    {
        GetStream() << " POINT-SIZE=" << "\"" << fontSize << "\"";
    }

    if (color && std::strlen(color) != 0)
    {
        GetStream() << " COLOR=\"" << color << "\" ";
    }

    if (face && std::strlen(face) != 0)
    {
        GetStream() << " FACE=\"" << face << "\" ";
    }

    GetStream() << ">";
}


HtmlFont::HtmlFont(std::ostream& stream)
    : HtmlFont(stream, -1, nullptr, nullptr)
{}

HtmlFont::~HtmlFont()
{
    GetStream() << "</FONT>";
}


DotAttributeSet::DotAttributeSet(std::ostream& stream)
    : DotBase(stream)
{
    GetStream() << "[";
}

DotAttributeSet::~DotAttributeSet()
{
    bool doSpace=false;
    for (auto&& attrib : m_Attributes)
    {
        if (doSpace)
        {
            GetStream() << " ";
        }

        GetStream() << attrib;
        doSpace=true;
    }

    GetStream() << "]";
}

DotAttributeSet & DotAttributeSet::AddAttribute(const std::string& name, const std::stringstream& value)
{
    std::stringstream ss;
    ss << name <<"=" << value.str();
    m_Attributes.push_back(ss.str());
    return *this;
}

DotAttributeSet & DotAttributeSet::AddAttribute(const std::string& name, int value)
{
    std::stringstream ss;
    ss << name <<"=" << value;
    m_Attributes.push_back(ss.str());
    return *this;
}

DotAttributeSet & DotAttributeSet::AddAttribute(const std::string& name, const std::string& value)
{
    std::stringstream ss;
    ss << name <<"=\"" << value << "\"";
    m_Attributes.push_back(ss.str());
    return *this;
}

DotEdge::DotEdge(std::ostream& stream, LayerGuid fromNodeId, LayerGuid toNodeId)
    : DotBase(stream)
{
    std::stringstream ss;
    ss << Indent(4) << fromNodeId << " -> " << toNodeId << " ";
    GetStream() << ss.str();

    m_Attributes = std::make_unique<DotAttributeSet>(stream);
}

DotEdge::~DotEdge()
{
    m_Attributes.reset(nullptr);
    GetStream() << ";" << std::endl;
}


NodeContent::NodeContent(std::ostream& stream)
    : DotBase(stream)
{
}

NodeContent & NodeContent::SetName(const std::string & name)
{
    m_Name = name;
    return *this;
}

NodeContent & NodeContent::AddContent(const std::string & content)
{
    m_Contents.push_back(content);
    return *this;
}

NodeContent::~NodeContent()
{
    std::stringstream ss;
    ss << "label=\"{" << m_Name;
    if (!m_Contents.empty())
    {
        ss << "|";
    }
    for (auto & content : m_Contents)
    {
        ss << Escape(content);
        ss << "\\l";
    }
    ss << "}\"";

    std::string s;
    try
    {
        // Coverity fix: std::stringstream::str() may throw an exception of type std::length_error.
        s = ss.str();
    }
    catch (const std::exception&) { } // Swallow any exception.

    GetStream() << s;
}

DotNode::DotNode(std::ostream& stream, LayerGuid nodeId, const char* label)
    : DotBase(stream)
{
    std::stringstream ss;
    ss << Indent(4) << nodeId;

    GetStream() << ss.str() << " ";

    m_Contents = std::make_unique<NodeContent>(stream);
    m_Attributes = std::make_unique<DotAttributeSet>(stream);

    if (std::strlen(label) != 0)
    {
        m_Contents->SetName(label);
    }
    else
    {
        m_Contents->SetName("<noname>");
    }
}

DotNode::~DotNode()
{
    m_Contents.reset(nullptr);
    m_Attributes.reset(nullptr);
    GetStream() << ";" << std::endl;
}


DotDefaults::DotDefaults(std::ostream& stream, const char* type)
    : DotBase(stream)
{
    std::stringstream ss;
    ss << Indent(4) << type;

    GetStream() << ss.str() << " ";
    m_Attributes = std::make_unique<DotAttributeSet>(stream);
}

DotDefaults::~DotDefaults()
{
    m_Attributes.reset(nullptr);
    GetStream() << ";" << std::endl;
}

DotGraph::DotGraph(std::ostream& stream, const char* name)
    : DotBase(stream)
{
    GetStream() << "digraph " << name << " {" << std::endl;
}

DotGraph::~DotGraph()
{
    GetStream() << "}" << std::endl;
}

} //namespace armnn


