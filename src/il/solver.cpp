#include "asserts.hpp"
#include "solver.hpp"
#include "logger.hpp"
#include "utils.hpp"

#include <fstream>
#include <stack>

#include <llvm/IR/Module.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/IR/InstrTypes.h>
#include <llvm/IR/Instruction.h>

#include <triton/llvmToTriton.hpp>

llvm::cl::opt<bool> save_branch_ast("solver-save-ast",
    llvm::cl::desc("Save branch ast into dot file on every branch."),
    llvm::cl::value_desc("flag"),
    llvm::cl::init(false),
    llvm::cl::Optional);

llvm::cl::opt<bool> print_branch_ast("solver-print-ast",
    llvm::cl::desc("Print branch ast on every branch."),
    llvm::cl::value_desc("flag"),
    llvm::cl::init(false),
    llvm::cl::Optional);

namespace il
{
std::vector<uint64_t> get_possible_targets(llvm::Value* ret, std::shared_ptr<Tracer> tracer)
{
    if (ret == nullptr)
    {
        logger::error("get_possible_targets argument got null argument.");
    }

    std::vector<uint64_t> targets;
    // Does not matter which arch we use.
    //
    triton::Context api(triton::arch::ARCH_X86_64);
    api.setAstRepresentationMode(triton::ast::representations::SMT_REPRESENTATION);

    if (auto inst = llvm::dyn_cast<llvm::Instruction>(ret))
    {
        if (inst->getOpcode() == llvm::Instruction::Or)
        {
            logger::warn("replacing or with add.");
            llvm::IRBuilder<> ir(inst);
            ret = ir.CreateAdd(inst->getOperand(0), inst->getOperand(1));
            inst->replaceAllUsesWith(ret);
        }
    }
    triton::ast::LLVMToTriton lifter(api);
    // Lift llvm into triton ast.
    //
    auto node = lifter.convert(ret);

    if (save_branch_ast)
    {
        static int solver_temp_names;
        std::fstream fd(fmt::format("branch-ast-{}.dot", solver_temp_names++), std::ios_base::out);
        if (fd.good())
            api.liftToDot(fd, node);
    }
    if (print_branch_ast)
    {
        logger::info("branch ast: {}", triton::ast::unroll(node));
    }
    // If constant.
    //
    if (!node->isSymbolized())
    {
        return { static_cast<uint64_t>(node->evaluate()) };
    }
    auto ast         = api.getAstContext();
    auto zero        = ast->bv(0, node->getBitvectorSize());
    auto constraints = ast->distinct(node, zero);

    while (true)
    {
        // Failsafe.
        //
        if (targets.size() > 2)
            break;

        auto model = api.getModel(constraints);

        if (model.empty())
            break;
        for (auto& [id, sym] : model)
            api.setConcreteVariableValue(api.getSymbolicVariable(id), sym.getValue());

        auto target = static_cast<uint64_t>(node->evaluate());
        targets.push_back(target);
        // Update constraints.
        //
        constraints = ast->land(constraints, ast->distinct(node, ast->bv(target, node->getBitvectorSize())));
    }

    if (targets.size() > 2) {
        logger::warn("get_possible_targets: failed to get jcc targets - too many results");
        return {};
    }
    else if (targets.empty()) {
        logger::warn("get_possible_targets: failed to get jcc targets - no results returned");
        // Solver fails to get results. Replace all ITE nodes with one of it's branch to reduce complexity and retry.
        using namespace triton::ast;
        std::vector<std::pair<SharedAbstractNode&, SharedAbstractNode>> ite_nodes;
        std::unordered_set<AbstractNode*> visited;
        std::stack<SharedAbstractNode*> worklist;

        auto get_parent_set = [](SharedAbstractNode& node) -> std::unordered_set<AbstractNode*> {
            std::unordered_set<AbstractNode*> res;
            for (auto& parent : node->getParents())
                res.insert(parent.get());
            return res;
        };

        worklist.push(&node);

        while (!worklist.empty()) {
            SharedAbstractNode& node = *worklist.top();
            worklist.pop();

            if (!visited.insert(node.get()).second) {
                continue;
            }

            // We know that OR with a value that has only one 0 or AND with a value that has only one 1 is the same as an ITE node.
            if (node->getType() == BVAND_NODE || node->getType() == BVOR_NODE) {
                for (const auto& operand : node->getChildren()) {
                    if (operand->getType() == BV_NODE) {
                        auto value = operand->evaluate();
                        auto size = operand->getBitvectorSize();
                        if (node->getType() == BVOR_NODE) value ^= operand->getBitvectorMask();
                        if ((value & (value - 1)) == 0) {
                            auto parents = get_parent_set(node);
                            node = ast->ite(ast->equal(node, ast->bv(operand->evaluate(), size)),
                                ast->bv(operand->evaluate(), size),
                                ast->bv(node->getType() == BVOR_NODE ? operand->getBitvectorMask() : 0, size));
                            node->setParent(parents);
                            break;
                        }
                    }
                }
            }

            if (node->getType() == ITE_NODE) {
                // Save the reference to the shared_ptr and also keep a copy of it so the node doesn't get destroyed.
                ite_nodes.push_back({ node, node });
                continue;
            }

            for (auto& r : node->getChildren()) {
                if (visited.find(r.get()) == visited.end()) {
                    worklist.push(&r);
                }
            }
        }

        if (ite_nodes.size() > 8) {
            logger::warn("get_possible_targets: too many ite nodes. failed to simplify");
            return {};
        }
        else if (ite_nodes.empty()) {
            return {};
        }

        logger::debug("trying to simplify ast");

        for (auto& [ref, original] : ite_nodes) {
            auto parents = get_parent_set(original);
            original->getChildren()[1]->setParent(parents);
            original->getChildren()[2]->setParent(parents);
        }

        std::set<uint64_t> simp_targets;
        std::set<uint64_t> invalid_targets;

        auto fork = tracer->fork();
        auto insn = std::get<vm::Jcc>(fork->step(step_t::execute_branch));
        uint64_t inst_cnt = fork->inst_cnt;

        // Random branch combinations might return invalid results; we need to filter them out.
        auto is_target_valid = [tracer, insn, inst_cnt](uint64_t target) -> bool {
            auto fork = tracer->fork();
            fork->write(fork->vsp(), target - (insn.direction() == vm::jcc_e::up ? 1 : -1) * 4);
            while (fork->inst_cnt < inst_cnt)
                fork->single_step();
            for (int j = 0; j < 10; j++)
            {
                auto inst = fork->single_step();
                if (op_mov_register_memory(inst))
                {
                    return true;
                }
            }
            return false;
        };

        for (int i = 0; i < 1 << ite_nodes.size(); i++) {
            if (simp_targets.size() > 2)
                break;

            for (int j = 0; j < ite_nodes.size(); j++) {
                ite_nodes[j].first = ite_nodes[j].second->getChildren()[(i & (1 << j)) ? 2 : 1];
                ite_nodes[j].first->init(true);
            }

            if (!node->isSymbolized()) {
                uint64_t target = static_cast<uint64_t>(node->evaluate());
                if (!simp_targets.contains(target) && !invalid_targets.contains(target)) {
                    if (is_target_valid(target))
                        simp_targets.insert(target);
                    else
                        invalid_targets.insert(target);
                }
                continue;
            }

            constraints = ast->distinct(node, zero);
            while (true)
            {
                if (simp_targets.size() > 2)
                    break;

                auto model = api.getModel(constraints);

                if (model.empty())
                    break;
                for (auto& [id, sym] : model)
                    api.setConcreteVariableValue(api.getSymbolicVariable(id), sym.getValue());

                auto target = static_cast<uint64_t>(node->evaluate());
                if (!simp_targets.contains(target) && !invalid_targets.contains(target)) {
                    if (is_target_valid(target))
                        simp_targets.insert(target);
                    else
                        invalid_targets.insert(target);
                }
                // Update constraints.
                //
                constraints = ast->land(constraints, ast->distinct(node, ast->bv(target, node->getBitvectorSize())));
            }
        }

        if (simp_targets.size() > 2) {
            logger::warn("get_possible_targets: simplified ast returns too many results");
            return {};
        }
        else if (simp_targets.empty()) {
            logger::warn("get_possible_targets: simplified ast returns no results");
            return {};
        }

        return std::vector<uint64_t>(simp_targets.begin(), simp_targets.end());
    }
    return targets;
}
};
