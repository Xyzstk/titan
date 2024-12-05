#include "asserts.hpp"
#include "tracer.hpp"
#include "logger.hpp"
#include "utils.hpp"

namespace variable
{
    static const std::string rsp          = "rsp";
    static const std::string vip          = "vip";
    static const std::string vip_fetch    = "[vip]";
    static const std::string vsp          = "vsp";
    static const std::string vsp_fetch    = "[vsp]";
    static const std::string vregs        = "vregs";
    static const std::string memory_fetch = "[memory]";
    static const std::string code_fetch   = "[code]"; // A symbol for possible ldr from code memory kept in main tracer for detecting jump tables.
};

// Match [vsp] + [vsp].
//
static bool match_add(const triton::ast::SharedAbstractNode& ast)
{
    if (ast->getType() == triton::ast::EXTRACT_NODE)
    {
        return match_add(ast->getChildren()[2]->getChildren()[1]);
    }
    return ast->getType() == triton::ast::BVADD_NODE
        && is_variable(ast->getChildren()[1], variable::vsp_fetch);
}

// Match `~[vsp] | ~[vsp]`.
//
static bool match_nand(const triton::ast::SharedAbstractNode& ast)
{
    // For nand_8 ast is following:
    // ((_ extract 15 0) (concat ((_ extract 63 8) (concat (_ bv0 48) [vsp])) (bvor (bvnot ((_ extract 7 0) [vsp])) (bvnot [vsp]))))
    //
    if (ast->getType() == triton::ast::EXTRACT_NODE)
    {
        return match_nand(ast->getChildren()[2]->getChildren()[1]);
    }
    return ast->getType() == triton::ast::BVOR_NODE
        && ast->getChildren()[1]->getType() == triton::ast::BVNOT_NODE
        && is_variable(ast->getChildren()[1]->getChildren()[0], variable::vsp_fetch);
}

// Match `~[vsp] & ~[vsp]`.
//
static bool match_nor(const triton::ast::SharedAbstractNode& ast)
{
    // For nor_8 ast is following:
    // ((_ extract 15 0) (concat ((_ extract 63 8) (concat (_ bv0 48) [vsp])) (bvand (bvnot ((_ extract 7 0) [vsp])) (bvnot [vsp]))))
    //
    if (ast->getType() == triton::ast::EXTRACT_NODE)
    {
        return match_nor(ast->getChildren()[2]->getChildren()[1]);
    }
    return ast->getType() == triton::ast::BVAND_NODE
        && ast->getChildren()[1]->getType() == triton::ast::BVNOT_NODE
        && is_variable(ast->getChildren()[1]->getChildren()[0], variable::vsp_fetch);
}

// Match `[vsp] >> ([vsp] & 0x3f)`.
//
static bool match_shr(const triton::ast::SharedAbstractNode& ast)
{
    // for shr
    if (ast->getType() == triton::ast::EXTRACT_NODE && ast->getChildren()[2]->getType() == triton::ast::CONCAT_NODE)
    {
        return ast->getChildren()[2]->getChildren()[1]->getType() == triton::ast::BVLSHR_NODE;
    }
    return ast->getType() == triton::ast::BVLSHR_NODE
        && ast->getChildren()[1]->getType() == triton::ast::BVAND_NODE
        && is_variable(ast->getChildren()[0], variable::vsp_fetch);
}

// Match `[vsp] << ([vsp] & 0x3f)`.
//
static bool match_shl(const triton::ast::SharedAbstractNode& ast)
{
    // for shl_8: ((_ extract 15 0) (concat ((_ extract 63 8) (concat (_ bv281474976710649 48) [vsp])) (bvshl ((_ extract 7 0) [vsp]) (bvand [vsp] (_ bv31 8)))))
    //
    if (ast->getType() == triton::ast::EXTRACT_NODE && ast->getChildren()[2]->getType() == triton::ast::CONCAT_NODE)
    {
        return ast->getChildren()[2]->getChildren()[1]->getType() == triton::ast::BVSHL_NODE;
    }
    return ast->getType() == triton::ast::BVSHL_NODE
        && ast->getChildren()[1]->getType() == triton::ast::BVAND_NODE
        && is_variable(ast->getChildren()[0], variable::vsp_fetch);
}

// Match `ror((([vsp]) << 32 | [vsp]), 0x0, 64)`
static bool match_shrd(const triton::ast::SharedAbstractNode& ast)
{
    return ast->getType() == triton::ast::EXTRACT_NODE
        && ast->getChildren()[2]->getType() == triton::ast::BVROR_NODE;
}

// Match `((_ extract 31 0) ((_ rotate_left 0) (concat [vsp] [vsp])))`.
//
static bool match_shld(const triton::ast::SharedAbstractNode& ast)
{
    return ast->getType() == triton::ast::EXTRACT_NODE
        && ast->getChildren()[2]->getType() == triton::ast::BVROL_NODE;
}

Tracer::Tracer(triton::arch::architecture_e arch) noexcept
    : Emulator(arch)
{
    physical_registers_count = (arch == triton::arch::ARCH_X86_64 ? 16 : 8);
}

Tracer::Tracer(Tracer const& other, bool fork_ast) noexcept
    : Emulator(other)
    , physical_registers_count{ other.physical_registers_count }
    , vip_register_name       { other.vip_register_name        }
    , vsp_register_name       { other.vsp_register_name        }
{
    if (fork_ast) {
        // Values are not passed by physical registers, so we only retain symbolized memory here.
        for (auto [addr, symexpr] : other.getSymbolicMemory()) {
            auto var = get_variable(collect_variables(symexpr->getAst(), true), variable::code_fetch);
            if (var.has_value()) {
                auto new_var = symbolic->symbolizeMemory(triton::arch::MemoryAccess(addr, triton::size::byte), variable::code_fetch);
                cached_states.emplace(new_var, other.cached_states.at(var.value()));
            }
        }
    }
}

uint64_t Tracer::vip() const
{
    return read(vip_register());
}

uint64_t Tracer::vsp() const
{
    return read(vsp_register());
}

const triton::arch::Register& Tracer::vip_register() const
{
    fassert(vip_register_name.has_value());
    return getRegister(vip_register_name.value());
}

const triton::arch::Register& Tracer::vsp_register() const
{
    fassert(vsp_register_name.has_value());
    return getRegister(vsp_register_name.value());
}

std::shared_ptr<Tracer> Tracer::fork(bool fork_ast) const noexcept
{
    return std::make_shared<Tracer>(*this, fork_ast);
}

vm::Instruction Tracer::step(step_t type)
{
    auto tracer = fork(true);

    if(type == step_t::stop_before_branch)
        logger::debug("Trying to decode vinst at 0x{:x}", tracer->rip());

    uint64_t code_fetch_rip = 0;

    if (auto vinsn_mb = tracer->process_instruction(type, &code_fetch_rip))
    {
        auto vinsn = vinsn_mb.value();
        if (vm::op_enter(vinsn))
        {
            vip_register_name = tracer->vip_register_name;
            vsp_register_name = tracer->vsp_register_name;
        }
        if (vm::op_branch(vinsn) && type == step_t::stop_before_branch)
            return vinsn;

        if (vm::op_jcc(vinsn))
        {
            vip_register_name = std::get<vm::Jcc>(vinsn).vip_register();
            vsp_register_name = std::get<vm::Jcc>(vinsn).vsp_register();
        }
        // Cicle tracer to the current rip value.
        //
        while (true)
        {
            auto insn = disassemble();
            if (insn.getAddress() == tracer->rip())
                break;
            if (insn.getAddress() == code_fetch_rip && std::holds_alternative<vm::Ldr>(vinsn)) {
                auto state = fork();
                execute(insn);
                auto var = symbolizeRegister(insn.operands[0].getConstRegister(), variable::code_fetch);
                cached_states.emplace(var, state);
            }
            else {
                execute(insn);
            }
        }
        return vinsn;
    }
    logger::error("Failed to process instruction");
}

std::optional<vm::Instruction> Tracer::process_instruction(step_t type, uint64_t* code_fetch_rip)
{
    if (!vip_register_name.has_value() || !vsp_register_name.has_value())
    {
        return process_vmenter();
    }
    // List of matched virtual instructions for this handler.
    //
    std::vector<vm::Instruction> vinsn;
    // List of executed instructions.
    //
    std::vector<triton::arch::Instruction> stream;
    // Symbolize bytecode and virtual stack.
    //
    symbolizeRegister(vip_register(), "vip");
    symbolizeRegister(vsp_register(), "vsp");
    symbolizeRegister(rsp_register(), "rsp");

    cache.clear();
    std::set<std::string> poped_registers;
    std::vector<vm::Pop>  poped_context;

    while (true)
    {
        auto insn = disassemble();
        // Handle memory write.
        //
        if (op_mov_memory_register(insn))
        {
            getSymbolicEngine()->initLeaAst(insn.operands[0].getMemory());
            if (auto vins = process_store(insn))
            {
                vinsn.push_back(std::move(vins.value()));
            }
        }
        // Handle pop register.
        //
        else if (op_pop_register(insn))
        {
            const auto& reg = insn.operands[0].getConstRegister();
            const auto name = reg.getName();
            if (!poped_registers.contains(name))
            {
                poped_context.push_back(vm::Pop(vm::PhysicalRegister(name), reg.getBitSize()));
                poped_registers.insert(std::move(name));
            }
        }
        else if (op_pop_flags(insn))
        {
            if (!poped_registers.contains("eflags"))
            {
                poped_context.push_back(vm::Pop(vm::PhysicalRegister("eflags"), 8 * ptrsize()));
                poped_registers.insert("eflags");
            }
        }
        // Build instruction semantics.
        //
        execute(insn);
        // Handle memory read.
        //
        if (op_mov_register_memory(insn))
        {
            if (auto vins = process_load(insn, code_fetch_rip))
            {
                vinsn.push_back(std::move(vins.value()));
            }
        }

        if (op_ret(insn) && poped_registers.size() == physical_registers_count)
        {
            stream.push_back(std::move(insn));
            break;
        }
        stream.push_back(std::move(insn));

        auto variables = collect_variables(getRegisterAst(rip_register()));

        if (has_variable(variables, variable::vip_fetch) ||
            has_variable(variables, variable::memory_fetch, variable::vsp_fetch))
        {
            break;
        }
        if (!rip())
        {
            break;
        }
    }

    if (vinsn.empty())
    {
        const auto variables = collect_variables(getRegisterAst(rip_register()));
        if (has_variable(variables, variable::memory_fetch, variable::vsp_fetch))
        {
            // Jcc handler.
            //
            auto comment   = get_variable(variables, variable::memory_fetch).value()->getComment();
            auto vip_reg   = getRegister(comment);
            auto vip_ast   = triton::ast::unroll(getRegisterAst(vip_reg));
            auto direction = vip_ast->getType() == triton::ast::BVADD_NODE ? vm::jcc_e::up : vm::jcc_e::down;
            std::string vsp_reg  = "";
            // Handle possible jump tables.
            //
            auto vsp_fetch_inst = lookup_instruction(get_variable(variables, variable::vsp_fetch).value());
            auto vip_fetch_variable = get_variable(collect_variables(getMemoryAst(vsp_fetch_inst.operands[1].getConstMemory()), true), variable::code_fetch);
            // Pick next handler and deduce vsp register. We know that the first instruction after jcc is pop so
            // first memory access should be access to vsp.
            auto tracer = fork();
            for (int i = 0; i < 10; i++)
            {
                auto insn = tracer->single_step();
                if (op_mov_register_memory(insn))
                {
                    vsp_reg = insn.operands[1].getConstMemory().getConstBaseRegister().getName();
                    if(!vip_fetch_variable.has_value() || type == step_t::execute_branch) return vm::Jcc(
                        direction,
                        vip_reg.getName(),
                        vsp_reg
                    );
                    break;
                }
            }
            if (!vip_fetch_variable.has_value()) {
                fassert("Failed to process jcc instruction.");
            }
            else {
                auto jtable_state = cached_states.at(vip_fetch_variable.value());
                auto jtable_fetch_inst = jtable_state->disassemble();
                auto value_size = jtable_fetch_inst.operands[1].getSize();
                auto base_register = jtable_fetch_inst.operands[1].getConstMemory().getConstBaseRegister();
                auto init_addr = jtable_state->read(base_register);
                // Check if the new vip is valid
                //
                if (vsp_reg == "") {
                    logger::warn("Jcc returns invalid vip.");
                    // This is likely to be caused by jump table out of bound.
                    // We iterate the memory where jump table possibly exists and check if any of the values returns a valid vip.
                    // Note that index is 32-bit value
                    for (int64_t i = -16; i <= 16; i++) {
                        if (!i) continue;
                        auto jtable_tracer = jtable_state->fork();
                        jtable_tracer->write(base_register, init_addr - value_size * (i & 0xFFFFFFFF));
                        // Cycle with the new possible jump table value till instruction count matches.
                        //
                        while(jtable_tracer->inst_cnt < inst_cnt)
                            jtable_tracer->single_step();
                        // Check if the new vip is valid.
                        //
                        for (int j = 0; j < 10; j++)
                        {
                            auto insn = jtable_tracer->single_step();
                            if (op_mov_register_memory(insn))
                            {
                                vsp_reg = insn.operands[1].getConstMemory().getConstBaseRegister().getName();
                                init_addr -= value_size * (i & 0xFFFFFFFF);
                                goto valid_jtable;
                            }
                        }
                    }
                    fassert("Failed to process jcc instruction.");
                }

            valid_jtable:
                // Find all valid jump table elements.
                //
                std::set<uint64_t> jcc_targets;
                uint64_t target;
                std::shared_ptr<Tracer> jtable_tracer = nullptr;
                uint64_t end_addr;
                int64_t index = 0;

            search_up:
                jtable_tracer = jtable_state->fork();
                jtable_tracer->write(base_register, init_addr + value_size * (++index));
                while (jtable_tracer->inst_cnt < inst_cnt)
                    jtable_tracer->single_step();
                target = jtable_tracer->read(vip_reg);
                for (int j = 0; j < 10; j++)
                {
                    auto insn = jtable_tracer->single_step();
                    if (op_mov_register_memory(insn))
                    {
                        jcc_targets.insert(target);
                        goto search_up;
                    }
                }
                end_addr = init_addr + value_size * index;
                index = 1;
            search_down:
                jtable_tracer = jtable_state->fork();
                jtable_tracer->write(base_register, init_addr + value_size * (--index));
                while (jtable_tracer->inst_cnt < inst_cnt)
                    jtable_tracer->single_step();
                target = jtable_tracer->read(vip_reg);
                for (int j = 0; j < 10; j++)
                {
                    auto insn = jtable_tracer->single_step();
                    if (op_mov_register_memory(insn))
                    {
                        jcc_targets.insert(target);
                        goto search_down;
                    }
                }

                logger::debug("Possible jump table found from 0x{:x} to 0x{:x}", init_addr + value_size * (index + 1), end_addr);
                return vm::Jcc(
                    direction,
                    vip_reg.getName(),
                    vsp_reg,
                    jcc_targets
                );
            }
        }
        else if (has_variable(variables, variable::vip_fetch) && ranges::any_of(stream, op_lea_rip))
        {
            // Jmp handler.
            //
            vinsn.push_back(vm::Jmp());
        }
        else if (poped_context.size() == physical_registers_count)
        {
            // Exit handler.
            //
            return vm::Exit(std::move(poped_context));
        }
    }
    if (vinsn.size() != 1)
    {
        for (const auto& insn : stream)
            logger::warn("{}", insn);
        return {};
    }
    return vinsn.at(0);
}

std::optional<vm::Instruction> Tracer::process_vmenter()
{
    // Save rsp for future lookup.
    //
    const auto rsp_value = rsp();
    // Symbolize initial context.
    //
    for (const auto& reg : regs())
    {
        symbolizeRegister(reg, reg.getName());
    }
    std::vector<triton::arch::Instruction> stream;
    // Execute vmenter. Collect virtual registers.
    //
    while (true)
    {
        auto insn = single_step();

        if (op_mov_register_register(insn))
        {
            const auto& r1 = insn.operands[0].getConstRegister();
            const auto& r2 = insn.operands[1].getConstRegister();

            if (r2 == rsp_register() && r1.getBitSize() == r2.getBitSize())
            {
                vsp_register_name = r1.getName();
            }
        }
        else if (op_mov_register_memory(insn))
        {
            const auto& r1 = insn.operands[0].getConstRegister();
            const auto& r2 = insn.operands[1].getConstMemory().getConstBaseRegister();
            if (r2 != rsp_register())
            {
                vip_register_name = r2.getName();
            }
            symbolizeRegister(r1, r1.getName());
        }
        stream.push_back(std::move(insn));
        if (isRegisterSymbolized(rip_register()))
            break;
    }

    if (!vip_register_name.has_value() || !vsp_register_name.has_value())
    {
        logger::warn("No virtual registers were found:");
        logger::warn("\tvip: {}", vip_register_name.has_value() ? "found" : "not found");
        logger::warn("\tvsp: {}", vsp_register_name.has_value() ? "found" : "not found");

        for (const auto& insn : stream)
            logger::warn("{}", insn);
        return {};
    }

    // Number of pushed physical registers on vmenter + 2 integers before vmenter and reloc at the end.
    //
    const auto context_size = physical_registers_count + 3;
    // Collect initial context.
    //
    std::vector<vm::Push> context;
    for (uint64_t addr = rsp_value - ptrsize(); addr >= rsp_value - context_size * ptrsize(); addr -= ptrsize())
    {
        triton::arch::MemoryAccess memory(addr, ptrsize());
        if (isMemorySymbolized(memory))
        {
            auto ast  = triton::ast::unroll(getMemoryAst(memory));
            auto size = ast->getBitvectorSize();
            fassert(ast->getType() == triton::ast::VARIABLE_NODE);
            context.push_back(vm::Push(vm::PhysicalRegister(to_variable(ast)->getAlias()), size));
        }
        else
        {
            // Match eflags since its not symbolic.
            //
            if (auto off = rsp_value - addr; off > 2 * ptrsize() && off < context_size * ptrsize())
            {
                context.push_back(vm::Push(vm::PhysicalRegister("eflags"), 8 * ptrsize()));
            }
            else
            {
                context.push_back(vm::Push(vm::Immediate(read<uint64_t>(memory)), 8 * ptrsize()));
            }
        }
    }
    if (context.size() != context_size)
        return {};
    return vm::Enter(std::move(context));
}

std::optional<vm::Instruction> Tracer::process_store(const triton::arch::Instruction& insn)
{
    const auto& mem    = insn.operands[0].getConstMemory();
    const auto& reg    = insn.operands[1].getConstRegister();
    auto mem_ast       = triton::ast::unroll(mem.getLeaAst());
    auto reg_ast       = triton::ast::unroll(getRegisterAst(reg));
    auto mem_variables = collect_variables(mem_ast);
    auto reg_variables = collect_variables(reg_ast);

    auto size = reg_ast->getBitvectorSize();

    if (reg_ast->getType() == triton::ast::EXTRACT_NODE && size == 16 && !has_variable(reg_variables, variable::vsp))
    {
        size = 8;
    }

    // movzx ax, byte ptr [vsp]
    // mov [vmregs + offset], ax
    //
    if (has_variable(mem_variables, variable::rsp, variable::vip_fetch) &&
        has_variable(reg_variables, variable::vsp_fetch))
    {
        auto write_off = read(mem.getConstIndexRegister());
        auto number    = write_off / ptrsize();
        auto offset    = write_off % ptrsize();
        auto original  = lookup_instruction(get_variable(reg_variables, variable::vsp_fetch).value());
        return vm::Pop(vm::VirtualRegister(number, offset), std::min(original.operands[1].getBitSize(), reg.getBitSize()));
    }
    if (has_variable(mem_variables, variable::vsp) &&
        has_variable(reg_variables, variable::vip_fetch))
    {
        return vm::Push(vm::Immediate(static_cast<uint64_t>(reg_ast->evaluate())), reg.getBitSize());
    }
    // mov ax, byte ptr [vmregs + offset]
    // mov [vsp], ax
    //
    if (has_variable(mem_variables, variable::vsp) &&
        has_variable(reg_variables, variable::vregs))
    {
        auto vreg = get_variable(reg_variables, variable::vregs).value();
        uint64_t index{};
        if (std::sscanf(vreg->getComment().c_str(), "0x%lx", &index) != 1)
        {
            logger::error("Failed to parse comment of push vreg instruction: {}", vreg->getComment());
        }
        auto number   = index / ptrsize();
        auto offset   = index % ptrsize();
        auto original = lookup_instruction(get_variable(reg_variables, variable::vregs).value());
        return vm::Push(vm::VirtualRegister(number, offset), std::min(original.operands[1].getBitSize(), reg.getBitSize()));
    }
    if (has_variable(mem_variables, variable::vsp) &&
        has_variable(reg_variables, variable::vsp))
    {
        return vm::Push(vm::VirtualStackPointer(), mem.getBitSize());
    }
    if (has_variable(mem_variables, variable::vsp_fetch) &&
        has_variable(reg_variables, variable::vsp_fetch))
    {
        return vm::Str(mem.getBitSize());
    }
    if (has_variable(mem_variables, variable::vsp) &&
        has_variable(reg_variables, variable::memory_fetch))
    {
        auto original = lookup_instruction(get_variable(reg_variables, variable::memory_fetch).value());
        return vm::Ldr(original.operands[1].getBitSize());
    }
    if (has_variable(mem_variables, variable::vsp) && match_add(reg_ast))
    {
        return vm::Add(size);
    }
    if (has_variable(mem_variables, variable::vsp) && match_nand(reg_ast))
    {
        return vm::Nand(size);
    }
    if (has_variable(mem_variables, variable::vsp) && match_nor(reg_ast))
    {
        return vm::Nor(size);
    }
    if (has_variable(mem_variables, variable::vsp) && match_shr(reg_ast))
    {
        return vm::Shr(size);
    }
    if (has_variable(mem_variables, variable::vsp) && match_shl(reg_ast))
    {
        return vm::Shl(size);
    }
    if (has_variable(mem_variables, variable::vsp) && match_shrd(reg_ast))
    {
        return vm::Shrd(size);
    }
    if (has_variable(mem_variables, variable::vsp) && match_shld(reg_ast))
    {
        return vm::Shld(size);
    }
    logger::warn("Failed to match store at 0x{:x}:", rip());
    logger::warn("\tmemory   AST: {}", mem_ast);
    logger::warn("\tregister AST: {}", reg_ast);
    return {};
}

std::optional<vm::Instruction> Tracer::process_load(const triton::arch::Instruction& insn, uint64_t* code_fetch_rip)
{
    const auto& reg = insn.operands[0].getConstRegister();
    const auto& mem = insn.operands[1].getConstMemory();
    const auto variables = collect_variables(mem.getLeaAst());

    if (has_variable(variables, variable::vip))
    {
        cache_instruction(insn, symbolizeRegister(reg, variable::vip_fetch));
    }
    else if (has_variable(variables, variable::vsp))
    {
        cache_instruction(insn, symbolizeRegister(reg, variable::vsp_fetch));

        if (vsp_register().isOverlapWith(reg))
        {
            return vm::Pop(vm::VirtualStackPointer(), mem.getBitSize());
        }
    }
    else if (has_variable(variables, variable::rsp, variable::vip_fetch))
    {
        // Set read offset as a comment to symbolic variable. It is used as vreg index in push vreg handler.
        //
        auto var = symbolizeRegister(reg, variable::vregs);
        var->setComment(fmt::format("0x{:x}", read(mem.getConstIndexRegister())));
        cache_instruction(insn, var);
    }
    else if (has_variable(variables, variable::vsp_fetch))
    {
        // Set memory operand register name as a comment to symbolic variable. It is used as new vip register in jcc handler.
        //
        auto var = symbolizeRegister(reg, variable::memory_fetch);
        var->setComment(fmt::format("{}", mem.getConstBaseRegister().getName()));
        cache_instruction(insn, var);
        // Find possible jump table fetch from code memory
        //
        if (mem.getAddress() >= stack_base && mem.getSize() >= triton::size::dword) *code_fetch_rip = insn.getAddress();
    }
    return {};
}

void Tracer::cache_instruction(triton::arch::Instruction insn, triton::engines::symbolic::SharedSymbolicVariable variable)
{
    cache.emplace(variable, insn);
}

const triton::arch::Instruction& Tracer::lookup_instruction(triton::engines::symbolic::SharedSymbolicVariable variable) const
{
    if (cache.find(variable) != cache.end())
        return cache.at(variable);
    logger::error("no instruction was found for {} variable", variable);
}
