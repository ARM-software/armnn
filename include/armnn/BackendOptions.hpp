//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "BackendId.hpp"
#include <cassert>

namespace armnn
{

struct BackendOptions;
using NetworkOptions = std::vector<BackendOptions>;

using ModelOptions = std::vector<BackendOptions>;

/// Struct for the users to pass backend specific options
struct BackendOptions
{
private:
    template<typename T>
    struct CheckAllowed
    {
        static const bool value = std::is_same<T, int>::value ||
                                  std::is_same<T, unsigned int>::value ||
                                  std::is_same<T, float>::value ||
                                  std::is_same<T, bool>::value ||
                                  std::is_same<T, std::string>::value ||
                                  std::is_same<T, const char*>::value;
    };
public:

    /// Very basic type safe variant
    class Var
    {

    public:
        /// Constructors
        explicit Var(int i) : m_Vals(i), m_Type(VarTypes::Integer) {};
        explicit Var(unsigned int u) : m_Vals(u), m_Type(VarTypes::UnsignedInteger) {};
        explicit Var(float f) : m_Vals(f), m_Type(VarTypes::Float) {};
        explicit Var(bool b) : m_Vals(b), m_Type(VarTypes::Boolean) {};
        explicit Var(const char* s) : m_Vals(s), m_Type(VarTypes::String) {};
        explicit Var(std::string s) : m_Vals(s), m_Type(VarTypes::String) {};

        /// Disallow implicit conversions from types not explicitly allowed below.
        template<typename DisallowedType>
        Var(DisallowedType)
        {
            static_assert(CheckAllowed<DisallowedType>::value, "Type is not allowed for Var<DisallowedType>.");
            assert(false && "Unreachable code");
        }

        /// Copy Construct
        Var(const Var& other)
            : m_Type(other.m_Type)
        {
            switch(m_Type)
            {
                case VarTypes::String:
                {
                    new (&m_Vals.s) std::string(other.m_Vals.s);
                    break;
                }
                default:
                {
                    DoOp(other, [](auto& a, auto& b)
                        {
                            a = b;
                        });
                    break;
                }
            }
        }

        /// Copy operator
        Var& operator=(const Var& other)
        {
            // Destroy existing string
            if (m_Type == VarTypes::String)
            {
                Destruct(m_Vals.s);
            }

            m_Type = other.m_Type;
            switch(m_Type)
            {
                case VarTypes::String:
                {

                    new (&m_Vals.s) std::string(other.m_Vals.s);
                    break;
                }
                default:
                {
                    DoOp(other, [](auto& a, auto& b)
                        {
                            a = b;
                        });
                    break;
                }
            }

            return *this;
        };

        /// Type getters
        bool IsBool() const { return m_Type == VarTypes::Boolean; }
        bool IsInt() const { return m_Type == VarTypes::Integer; }
        bool IsUnsignedInt() const { return m_Type == VarTypes::UnsignedInteger; }
        bool IsFloat() const { return m_Type == VarTypes::Float; }
        bool IsString() const { return m_Type == VarTypes::String; }

        /// Value getters
        bool AsBool() const { assert(IsBool()); return m_Vals.b; }
        int AsInt() const { assert(IsInt()); return m_Vals.i; }
        unsigned int AsUnsignedInt() const { assert(IsUnsignedInt()); return m_Vals.u; }
        float AsFloat() const { assert(IsFloat()); return m_Vals.f; }
        std::string AsString() const { assert(IsString()); return m_Vals.s; }

        /// Destructor
        ~Var()
        {
            DoOp(*this, [this](auto& a, auto&)
                {
                    Destruct(a);
                });
        }
    private:
        template<typename Func>
        void DoOp(const Var& other, Func func)
        {
            if (other.IsBool())
            {
                func(m_Vals.b, other.m_Vals.b);
            }
            else if (other.IsInt())
            {
                func(m_Vals.i, other.m_Vals.i);
            }
            else if (other.IsUnsignedInt())
            {
                func(m_Vals.u, other.m_Vals.u);
            }
            else if (other.IsFloat())
            {
                func(m_Vals.f, other.m_Vals.f);
            }
            else if (other.IsString())
            {
                func(m_Vals.s, other.m_Vals.s);
            }
        }

        template<typename Destructable>
        void Destruct(Destructable& d)
        {
            if (std::is_destructible<Destructable>::value)
            {
                d.~Destructable();
            }
        }

    private:
        /// Types which can be stored
        enum class VarTypes
        {
            Boolean,
            Integer,
            Float,
            String,
            UnsignedInteger
        };

        /// Union of potential type values.
        union Vals
        {
            int i;
            unsigned int u;
            float f;
            bool b;
            std::string s;

            Vals(){}
            ~Vals(){}

            explicit Vals(int i) : i(i) {};
            explicit Vals(unsigned int u) : u(u) {};
            explicit Vals(float f) : f(f) {};
            explicit Vals(bool b) : b(b) {};
            explicit Vals(const char* s) : s(std::string(s)) {}
            explicit Vals(std::string s) : s(s) {}
       };

        Vals m_Vals;
        VarTypes m_Type;
    };

    struct BackendOption
    {
    public:
        BackendOption(std::string name, bool value)
            : m_Name(name), m_Value(value)
        {}
        BackendOption(std::string name, int value)
            : m_Name(name), m_Value(value)
        {}
        BackendOption(std::string name, unsigned int value)
                : m_Name(name), m_Value(value)
        {}
        BackendOption(std::string name, float value)
            : m_Name(name), m_Value(value)
        {}
        BackendOption(std::string name, std::string value)
            : m_Name(name), m_Value(value)
        {}
        BackendOption(std::string name, const char* value)
            : m_Name(name), m_Value(value)
        {}

        template<typename DisallowedType>
        BackendOption(std::string, DisallowedType)
            : m_Value(0)
        {
            static_assert(CheckAllowed<DisallowedType>::value, "Type is not allowed for BackendOption.");
            assert(false && "Unreachable code");
        }

        BackendOption(const BackendOption& other) = default;
        BackendOption(BackendOption&& other) = default;
        BackendOption& operator=(const BackendOption& other) = default;
        BackendOption& operator=(BackendOption&& other) = default;
        ~BackendOption() = default;

        std::string GetName() const   { return m_Name; }
        Var GetValue() const          { return m_Value; }

    private:
        std::string m_Name;         ///< Name of the option
        Var         m_Value;        ///< Value of the option. (Bool, int, Float, String)
    };

    explicit BackendOptions(BackendId backend)
        : m_TargetBackend(backend)
    {}

    BackendOptions(BackendId backend, std::initializer_list<BackendOption> options)
        : m_TargetBackend(backend)
        , m_Options(options)
    {}

    BackendOptions(const BackendOptions& other) = default;
    BackendOptions(BackendOptions&& other) = default;
    BackendOptions& operator=(const BackendOptions& other) = default;
    BackendOptions& operator=(BackendOptions&& other) = default;

    void AddOption(BackendOption&& option)
    {
        m_Options.push_back(option);
    }

    void AddOption(const BackendOption& option)
    {
        m_Options.push_back(option);
    }

    const BackendId& GetBackendId() const noexcept { return m_TargetBackend; }
    size_t GetOptionCount() const noexcept { return m_Options.size(); }
    const BackendOption& GetOption(size_t idx) const { return m_Options[idx]; }

private:
    /// The id for the backend to which the options should be passed.
    BackendId m_TargetBackend;

    /// The array of options to pass to the backend context
    std::vector<BackendOption> m_Options;
};


template <typename F>
void ParseOptions(const std::vector<BackendOptions>& options, BackendId backend, F f)
{
    for (auto optionsGroup : options)
    {
        if (optionsGroup.GetBackendId() == backend)
        {
            for (size_t i=0; i < optionsGroup.GetOptionCount(); i++)
            {
                const BackendOptions::BackendOption option = optionsGroup.GetOption(i);
                f(option.GetName(), option.GetValue());
            }
        }
    }
}

} //namespace armnn
