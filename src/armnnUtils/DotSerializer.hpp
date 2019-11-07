//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/Types.hpp>

#include <ostream>
#include <vector>
#include <memory>

namespace armnn
{

class DotBase
{
public:
    explicit DotBase(std::ostream& stream)
        : m_Stream(stream) {}

    std::ostream& GetStream() { return m_Stream; }

private:
    std::ostream& m_Stream;
};

class HtmlSection : public DotBase
{
public:
    explicit HtmlSection(std::ostream& stream)
        : DotBase(stream) { GetStream() << "<";}
    ~HtmlSection() { GetStream() << ">"; }
};

class HtmlSimpleTag : public DotBase
{
public:
    explicit HtmlSimpleTag(std::ostream& stream, const char* name)
        : DotBase(stream)
        , m_Name(name){ GetStream() << "<" << m_Name << ">"; }
    ~HtmlSimpleTag() { GetStream() << "</" << m_Name << ">"; }

private:
    const char* m_Name;
};

class HtmlBold : public HtmlSimpleTag
{
public:
    explicit HtmlBold(std::ostream &stream)
        : HtmlSimpleTag(stream, "B") {}
};

class HtmlFont : public DotBase
{
public:
    explicit HtmlFont(std::ostream& stream, int fontSize, const char* color, const char* face);
    explicit HtmlFont(std::ostream& stream);
    ~HtmlFont();
};

class DotAttributeSet : public DotBase
{
public:
    explicit DotAttributeSet(std::ostream& stream);
    ~DotAttributeSet();

    DotAttributeSet & AddAttribute(const std::string& name, const std::stringstream& value);
    DotAttributeSet & AddAttribute(const std::string& name, int value);
    DotAttributeSet & AddAttribute(const std::string& name, const std::string& value);
private:
    std::vector<std::string> m_Attributes;
};

class DotEdge : public DotBase
{
public:
    explicit DotEdge(std::ostream& stream, LayerGuid fromNodeId, LayerGuid toNodeId);
    ~DotEdge();

    DotAttributeSet& GetAttributeSet() { return *m_Attributes.get(); }
private:
    std::unique_ptr<DotAttributeSet> m_Attributes;
};

class NodeContent : public DotBase
{
public:
    explicit NodeContent(std::ostream& stream);
    NodeContent & SetName(const std::string & name);
    NodeContent & AddContent(const std::string & content);

    ~NodeContent();
private:
    std::string m_Name;
    std::vector<std::string> m_Contents;
};

class DotNode : public DotBase
{
public:
    explicit DotNode(std::ostream& stream, LayerGuid nodeId, const char* label);
    ~DotNode();

    NodeContent& GetContents()         { return *m_Contents.get(); }
    DotAttributeSet& GetAttributeSet() { return *m_Attributes.get(); }
private:
    std::unique_ptr<NodeContent>     m_Contents;
    std::unique_ptr<DotAttributeSet> m_Attributes;
};

class DotDefaults : public DotBase
{
public:
    explicit DotDefaults(std::ostream& stream, const char* type);
    ~DotDefaults();

    DotAttributeSet& GetAttributeSet() { return *m_Attributes.get(); }
private:
    std::unique_ptr<DotAttributeSet> m_Attributes;
};

class DotGraph : public DotBase
{
public:
    explicit DotGraph(std::ostream& stream, const char* name);
    ~DotGraph();
private:
};

} //namespace armnn
