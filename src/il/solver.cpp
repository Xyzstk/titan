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
std::vector<uint64_t> get_possible_targets(llvm::Value* ret)
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

        worklist.push(&node);

        while (!worklist.empty()) {
            SharedAbstractNode& ast = *worklist.top();
            worklist.pop();

            if (!visited.insert(ast.get()).second) {
                continue;
            }

            if (ast->getType() == ITE_NODE) {
                // Save the reference to the shared_ptr and also keep a copy of it so the node doesn't get destroyed.
                ite_nodes.push_back({ ast, ast });
                continue;
            }

            for (auto& r : ast->getChildren()) {
                if (visited.find(r.get()) == visited.end()) {
                    worklist.push(&r);
                }
            }
        }

        if (ite_nodes.size() > 4) {
            logger::warn("get_possible_targets: too many ite nodes. failed to simplify");
            return {};
        }
        else if (ite_nodes.empty()) {
            return {};
        }

        logger::debug("trying to simplify ast");

        std::set<uint64_t> simp_targets;

        for (int i = 0; i < 1 << ite_nodes.size(); i++) {
            if (simp_targets.size() > 2)
                break;

            for (int j = 0; j < ite_nodes.size(); j++)
                ite_nodes[j].first = ite_nodes[j].second->getChildren()[(i & (1 << j)) ? 2 : 1];

            if (collect_variables(node).empty()) {
                simp_targets.insert(static_cast<uint64_t>(node->evaluate()));
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
                simp_targets.insert(target);
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
