#pragma once

#include "emulator.hpp"
#include "vm/instruction.hpp"

enum class step_t
{
    stop_before_branch,
    execute_branch
};

struct Tracer final : Emulator
{
    Tracer(triton::arch::architecture_e arch) noexcept;
    Tracer(Tracer const& other, bool fork_ast) noexcept;

    std::shared_ptr<Tracer> fork(bool fork_ast = false) const noexcept;

    uint64_t vip() const;
    uint64_t vsp() const;

    const triton::arch::Register& vip_register() const;
    const triton::arch::Register& vsp_register() const;

    std::unordered_map<triton::engines::symbolic::SharedSymbolicVariable, std::shared_ptr<Tracer>> cached_states;

    vm::Instruction step(step_t type);

private:
    std::optional<vm::Instruction> process_instruction(step_t type, uint64_t* code_fetch_rip);
    std::optional<vm::Instruction> process_vmenter();
    std::optional<vm::Instruction> process_store(const triton::arch::Instruction& insn);
    std::optional<vm::Instruction> process_load (const triton::arch::Instruction& insn, uint64_t* code_fetch_rip);

    void cache_instruction(triton::arch::Instruction insn, triton::engines::symbolic::SharedSymbolicVariable variable);
    const triton::arch::Instruction& lookup_instruction(triton::engines::symbolic::SharedSymbolicVariable variable) const;

    // Context size based on architecture.
    //
    size_t physical_registers_count;

    std::optional<std::string> vip_register_name;
    std::optional<std::string> vsp_register_name;

    std::unordered_map<triton::engines::symbolic::SharedSymbolicVariable, triton::arch::Instruction> cache;
};
