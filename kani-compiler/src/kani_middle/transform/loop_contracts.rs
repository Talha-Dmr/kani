// Copyright Kani Contributors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! This module contains code related to the MIR-to-MIR pass to enable loop contracts.
//!

use super::TransformPass;
use crate::kani_middle::KaniAttributes;
use crate::kani_middle::codegen_units::CodegenUnit;
use crate::kani_middle::kani_functions::KaniModel;
use crate::kani_middle::transform::TransformationType;
use crate::kani_middle::transform::body::{
    InsertPosition, MutableBody, SourceInstruction, synthetic_source_info,
};
use crate::kani_queries::QueryDb;
use crate::rustc_public::CrateDef;
use itertools::Itertools;
use rustc_middle::ty::TyCtxt;
use rustc_public::mir::mono::Instance;
use rustc_public::mir::{
    AggregateKind, BasicBlock, BasicBlockIdx, Body, BorrowKind, ConstOperand, Mutability,
    NonDivergingIntrinsic, Operand, Place, ProjectionElem, RawPtrKind, Rvalue, Statement,
    StatementKind, SwitchTargets, Terminator, TerminatorKind, VarDebugInfoContents, WithRetag,
};
use rustc_public::ty::{FnDef, GenericArgKind, MirConst, RigidTy, TyKind, UintTy};
use rustc_span::Symbol;
use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt::Debug;

#[derive(Debug, Default, Clone)]
pub struct LoopContractPass {
    /// Cache KaniRunContract function used to implement contracts.
    run_contract_fn: Option<FnDef>,
    /// The map from original loop head to the new loop latch.
    /// We use this map to redirect all original loop latches to a new single loop latch.
    new_loop_latches: HashMap<usize, usize>,
}

impl TransformPass for LoopContractPass {
    /// The type of transformation that this pass implements.
    fn transformation_type() -> TransformationType
    where
        Self: Sized,
    {
        TransformationType::Stubbing
    }

    fn is_enabled(&self, query_db: &QueryDb) -> bool
    where
        Self: Sized,
    {
        query_db.args().unstable_features.contains(&"loop-contracts".to_string())
    }

    /// Run a transformation pass on the whole codegen unit.
    ///
    /// This pass will perform the following operations:
    /// 1. Replace the body of `kani_register_loop_contract` by `kani::internal::run_contract_fn`
    ///    to invoke the closure.
    ///
    /// 2. Transform loops with contracts from
    ///    ```ignore
    ///    bb_idx: {
    ///         loop_head_stmts
    ///         _v = kani_register_loop_contract(move args) -> [return: terminator_target];
    ///    }
    ///
    ///    ...
    ///    loop_body_blocks
    ///    ...
    ///
    ///    loop_latch_block: {
    ///         loop_latch_stmts
    ///         goto -> bb_idx;
    ///    }
    ///    ```
    ///    to blocks
    ///    ```ignore
    ///    bb_idx: {
    ///         loop_head_stmts
    ///         _v = true
    ///         goto -> terminator_target
    ///    }
    ///
    ///    ...
    ///    loop_body_blocks
    ///    ...
    ///
    ///    loop_latch_block: {
    ///         loop_latch_stmts
    ///         goto -> bb_new_loop_latch;
    ///    }
    ///
    ///    bb_new_loop_latch: {
    ///         loop_head_body
    ///         _v = kani_register_loop_contract(move args) -> [return: terminator_target];
    ///    }
    ///    ```
    fn transform(&mut self, tcx: TyCtxt, body: Body, instance: Instance) -> (bool, Body) {
        self.new_loop_latches = HashMap::new();
        match instance.ty().kind().rigid().unwrap() {
            RigidTy::FnDef(_func, args) => {
                if KaniAttributes::for_instance(tcx, instance).fn_marker()
                    == Some(Symbol::intern("kani_register_loop_contract"))
                {
                    // Replace the body of the register function with `run_contract_fn`'s.
                    let run = Instance::resolve(self.run_contract_fn.unwrap(), args).unwrap();
                    (true, run.body().unwrap())
                } else {
                    self.transform_body_with_loop(tcx, body)
                }
            }
            RigidTy::Closure(_, _) => self.transform_body_with_loop(tcx, body),
            _ => {
                /* static variables case */
                (false, body)
            }
        }
    }
}

impl LoopContractPass {
    pub fn new(_tcx: TyCtxt, queries: &QueryDb, unit: &CodegenUnit) -> LoopContractPass {
        if !unit.harnesses.is_empty() {
            let run_contract_fn =
                queries.kani_functions().get(&KaniModel::RunLoopContract.into()).copied();
            assert!(run_contract_fn.is_some(), "Failed to find Kani run contract function");
            LoopContractPass { run_contract_fn, new_loop_latches: HashMap::new() }
        } else {
            // If reachability mode is PubFns or Tests, we just remove any contract logic.
            // Note that in this path there is no proof harness.
            LoopContractPass::default()
        }
    }

    /// Generate the body of loop head block by dropping all statements
    /// except for `StorageLive` and `StorageDead`.
    fn get_loop_head_block(&self, block: &BasicBlock) -> BasicBlock {
        let new_stmts: Vec<Statement> = block
            .statements
            .iter()
            .filter(|stmt| {
                matches!(stmt.kind, StatementKind::StorageLive(_) | StatementKind::StorageDead(_))
            })
            .cloned()
            .collect();
        BasicBlock { statements: new_stmts, terminator: block.terminator.clone() }
    }

    /// Remove `StorageDead closure_var` to avoid invariant closure becoming dead.
    fn make_invariant_closure_alive(&self, body: &mut MutableBody, bb_idx: usize) {
        let mut stmts = body.blocks()[bb_idx].statements.clone();
        if stmts.is_empty() || !matches!(stmts[0].kind, StatementKind::StorageDead(_)) {
            unreachable!(
                "The assumptions for loop-contracts transformation are violated by some other transformation. \
            Please report github.com/model-checking/kani/issues/new?template=bug_report.md"
            );
        }
        stmts.remove(0);
        body.replace_statements(&SourceInstruction::Terminator { bb: bb_idx }, stmts);
    }

    //Get all user defined variables in the function (not the compiler-generated ones)
    //Note that there might be user defined variables with the same user defined names,
    //but they are all have different MIR-generated names.
    fn get_user_defined_variables(&self, body: &MutableBody) -> Vec<usize> {
        let mut user_vars = Vec::new();

        // Iterate through all locals
        for (idx, _) in body.locals().iter().enumerate() {
            // Skip the return place (local 0)
            if idx == 0 {
                continue;
            }

            // Check if this is a user-defined variable (not a compiler temp)
            let is_user_defined = body.var_debug_info().iter().any(|info| {
            matches!(&info.value, VarDebugInfoContents::Place(place) if place.local == idx)
        });

            if is_user_defined {
                user_vars.push(idx);
            }
        }

        user_vars
    }

    // Get the list of tuples:
    // (firstpat, the block_id where firstpat get assigned, the corresponding nthpat, the block_id where nthpat get assigned)
    fn get_first_pats_and_nth_pats(&self, body: &MutableBody) -> Vec<(usize, usize, usize, usize)> {
        let mut first_pats_and_nth_pats: Vec<(usize, usize, usize, usize)> = Vec::new();
        let mut current_firstpat = 0;
        let mut current_firstpat_pos = 0;
        for (blockid, block) in body.blocks().iter().enumerate() {
            if let TerminatorKind::Call {
                func: terminator_func,
                args: _,
                destination: dest,
                target: _,
                unwind: _,
            } = &block.terminator.kind
            {
                // Get the function signature of the terminator call.
                let Some(RigidTy::FnDef(fn_def, _)) = terminator_func
                    .ty(body.locals())
                    .ok()
                    .map(|fn_ty| fn_ty.kind().rigid().unwrap().clone())
                else {
                    continue;
                };
                if fn_def.name() == "kani::KaniIter::first" {
                    current_firstpat = dest.local;
                    current_firstpat_pos = blockid;
                }
                if fn_def.name() == "kani::KaniIter::nth" && current_firstpat != 0 {
                    first_pats_and_nth_pats.push((
                        current_firstpat,
                        dest.local,
                        current_firstpat_pos,
                        blockid,
                    ));
                    current_firstpat = 0
                }
            }
        }
        first_pats_and_nth_pats
    }

    // This Vec includes the user defined variables together with the tuple-typed variable
    // thst store the return of "kani::KaniIter::first" function
    fn get_storage_moving_variables(&self, body: &MutableBody) -> Vec<usize> {
        let first_nth_list = self.get_first_pats_and_nth_pats(body);
        let mut moving_vars = self.get_user_defined_variables(body);
        for (firstvar, _, _, _) in first_nth_list {
            if !moving_vars.contains(&firstvar) {
                moving_vars.push(firstvar);
            }
        }
        moving_vars
    }

    // Return the same Terminator with new destination
    fn terminator_of_new_destination(old: Terminator, new_destination_local: usize) -> Terminator {
        let mut terminator = old.clone();
        if let TerminatorKind::Call {
            func: terminator_func,
            args: terminator_args,
            destination: old_destination,
            target: terminator_target,
            unwind: terminator_unwind,
        } = &terminator.kind
        {
            let mut new_destination = old_destination.clone();
            new_destination.local = new_destination_local;
            terminator.kind = TerminatorKind::Call {
                func: terminator_func.clone(),
                args: terminator_args.clone(),
                destination: new_destination,
                target: *terminator_target,
                unwind: *terminator_unwind,
            };
        }
        terminator
    }

    // Replace the "firstpat" vars with its corresponding "nthpat" vars
    // See the comments in kani/library/kani_macros/src/sysroot/loop_contracts/mod.rs
    fn replace_first_pat_by_nth_pat(&self, body: &mut MutableBody) {
        let first_nth_list = self.get_first_pats_and_nth_pats(body);
        for (firstvar, nthvar, first_blockid, nth_blockid) in first_nth_list {
            // Replace firstpat by nthpat in the destination of "kani::KaniIter::first" function call
            let old_terminator = body.blocks()[first_blockid].terminator.clone();
            let new_terminator = Self::terminator_of_new_destination(old_terminator, nthvar);
            body.replace_terminator(
                &SourceInstruction::Terminator { bb: first_blockid },
                new_terminator,
            );
            let span = body.blocks()[first_blockid].statements.first().unwrap().source_info.span;
            // Add the StorageLive(nthpat) statement at the begining of the same block
            let storagelive_stmt = Statement {
                kind: StatementKind::StorageLive(nthvar),
                source_info: synthetic_source_info(span),
            };
            body.insert_stmt(
                storagelive_stmt,
                &mut SourceInstruction::Statement { idx: 0, bb: first_blockid },
                InsertPosition::Before,
            );

            // Remove the StorageLive(firstpat) statement in the same block if any
            let mut storageliveid = None;
            for (id, stmt) in body.blocks()[first_blockid].statements.iter().enumerate() {
                if let StatementKind::StorageLive(local) = &stmt.kind
                    && *local == firstvar
                {
                    storageliveid = Some(id)
                }
            }
            if let Some(id) = storageliveid {
                body.remove_stmt(first_blockid, id);
            }

            // Construct the HashMap of the firstpat projections with  nthpat projections
            let firstprj_stmts_copy = body.blocks()[first_blockid + 1].statements.clone();
            let nthprj_stmts_copy = body.blocks()[nth_blockid + 1].statements.clone();
            let mut firstprj_nthprj: HashMap<usize, usize> = HashMap::new();
            firstprj_nthprj.insert(firstvar, nthvar);

            // Only user variables (pattern bindings) are paired up with their nthpat
            // counterparts. Compiler-generated temporaries (e.g. the deref temporaries
            // that destructuring patterns like `&x` introduce) must not be paired:
            // their assignments are kept in place (with the rvalue redirected to
            // nthvar) so that later statements reading them remain well-defined.
            // Note: before rust-lang/rust#145513 such temporaries were assigned via
            // `Rvalue::CopyForDeref` and thus never matched the `Rvalue::Use` pattern
            // below; nowadays they are plain `Rvalue::Use(Operand::Copy)` assignments.
            let user_vars = self.get_user_defined_variables(body);
            for fstmt in firstprj_stmts_copy.iter() {
                if let StatementKind::Assign(fprjplace, frval) = &fstmt.kind
                    && let Rvalue::Use(Operand::Copy(firstpatplace), _) = frval
                    && firstpatplace.local == firstvar
                    && user_vars.contains(&fprjplace.local)
                {
                    let firstprj = fprjplace.local;
                    for istmt in nthprj_stmts_copy.iter() {
                        if let StatementKind::Assign(iprjplace, irval) = &istmt.kind
                            && let Rvalue::Use(Operand::Copy(nthpatplace), _) = irval
                            && nthpatplace.local == nthvar
                            && nthpatplace.projection == firstpatplace.projection
                        {
                            let nthprj = iprjplace.local;
                            firstprj_nthprj.insert(firstprj, nthprj);
                            break;
                        }
                    }
                }
            }

            // Replace the firstpat (and its projections) with nthpat (and its projections)
            // in the places where they may get involved, which includes, the comments in code.
            // First, in the block that  "kani::KaniIter::first" is called
            let mut new_stmts: Vec<Statement> = Vec::new();
            for stmt in firstprj_stmts_copy.iter() {
                let mut new_stmt = stmt.clone();

                match &stmt.kind {
                    // The StorageLive statements of the projections
                    // There might be some without StorageLive statements
                    // So we just remove them and add a new one for each nthpat projection later
                    StatementKind::StorageLive(local) => {
                        if !firstprj_nthprj.contains_key(local) {
                            new_stmts.push(new_stmt);
                        }
                    }
                    // The assign statements of the projections
                    StatementKind::Assign(fprjplace, frval) => {
                        match frval {
                            Rvalue::Use(Operand::Copy(firstpatplace), _) => {
                                if firstpatplace.local == firstvar {
                                    let mut nthpatplace = firstpatplace.clone();
                                    nthpatplace.local = nthvar;
                                    let newrval =
                                        Rvalue::Use(Operand::Copy(nthpatplace), WithRetag::No);
                                    if let Some(nthprj) = firstprj_nthprj.get(&fprjplace.local) {
                                        // A user pattern binding: redirect the assignment to
                                        // the corresponding nthpat projection variable.
                                        new_stmt.kind = StatementKind::Assign(
                                            Place {
                                                local: *nthprj,
                                                projection: fprjplace.projection.clone(),
                                            },
                                            newrval,
                                        );
                                    } else {
                                        // A compiler-generated temporary (e.g. a deref
                                        // temporary, `Rvalue::CopyForDeref` before
                                        // rust-lang/rust#145513): keep the assigned place and
                                        // only redirect the rvalue to read from nthvar, so
                                        // that following statements dereferencing the
                                        // temporary keep working.
                                        new_stmt.kind =
                                            StatementKind::Assign(fprjplace.clone(), newrval);
                                    }
                                }
                            }
                            Rvalue::CopyForDeref(firstpatplace) => {
                                if firstpatplace.local == firstvar {
                                    let mut nthpatplace = firstpatplace.clone();
                                    nthpatplace.local = nthvar;
                                    let newrval = Rvalue::CopyForDeref(nthpatplace);
                                    new_stmt.kind =
                                        StatementKind::Assign(fprjplace.clone(), newrval);
                                }
                            }
                            _ => {
                                if let Some(nthprj) = firstprj_nthprj.get(&fprjplace.local) {
                                    new_stmt.kind = StatementKind::Assign(
                                        Place {
                                            local: *nthprj,
                                            projection: fprjplace.projection.clone(),
                                        },
                                        frval.clone(),
                                    )
                                }
                            }
                        }
                        new_stmts.push(new_stmt);
                    }
                    _ => new_stmts.push(new_stmt),
                }
            }

            body.replace_statements(
                &SourceInstruction::Statement { idx: 0, bb: first_blockid + 1 },
                new_stmts,
            );

            // Second, in the loophead block right after that
            let loophead_stmts_copy = body.blocks()[first_blockid + 2].statements.clone();
            let mut new_loophead_stmts = Vec::new();
            if let StatementKind::Assign(_, Rvalue::Ref(_, _, Place { local: closurelocal, .. })) =
                loophead_stmts_copy.last().unwrap().kind
            {
                for stmt in loophead_stmts_copy.iter() {
                    // In the Operands of the loop invariant closure
                    if let StatementKind::Assign(lhs, Rvalue::Aggregate(aggrkind, operands)) =
                        &stmt.kind
                        && lhs.local == closurelocal
                    {
                        let mut new_operands = Vec::new();
                        for operand in operands.iter() {
                            if let Operand::Move(Place { local: operandlocal, projection: proj }) =
                                operand
                                && let Some(nthprj) = firstprj_nthprj.get(operandlocal)
                            {
                                let new_operand = Operand::Move(Place {
                                    local: *nthprj,
                                    projection: proj.clone(),
                                });
                                new_operands.push(new_operand);
                            } else if let Operand::Copy(Place {
                                local: operandlocal,
                                projection: proj,
                            }) = operand
                                && let Some(nthprj) = firstprj_nthprj.get(operandlocal)
                            {
                                let new_operand = Operand::Copy(Place {
                                    local: *nthprj,
                                    projection: proj.clone(),
                                });
                                new_operands.push(new_operand);
                            } else {
                                new_operands.push(operand.clone());
                            }
                        }
                        let new_rval = Rvalue::Aggregate(aggrkind.clone(), new_operands);
                        new_loophead_stmts.push(Statement {
                            kind: StatementKind::Assign(lhs.clone(), new_rval),
                            source_info: synthetic_source_info(stmt.source_info.span),
                        });
                    } else if let StatementKind::Assign(
                        lhs,
                        Rvalue::Ref(region, borrowkind, Place { local: firstlocal, projection }),
                    ) = &stmt.kind
                         // In the borrow statements 
                        && let Some(nthlocal) = firstprj_nthprj.get(firstlocal)
                    {
                        let new_rval = Rvalue::Ref(
                            region.clone(),
                            *borrowkind,
                            Place { local: *nthlocal, projection: projection.clone() },
                        );
                        new_loophead_stmts.push(Statement {
                            kind: StatementKind::Assign(lhs.clone(), new_rval),
                            source_info: synthetic_source_info(stmt.source_info.span),
                        });
                    } else {
                        new_loophead_stmts.push(stmt.clone());
                    }
                }
            } else {
                panic!("not a loop head")
            }

            body.replace_statements(
                &SourceInstruction::Statement { idx: 0, bb: first_blockid + 2 },
                new_loophead_stmts,
            );

            // Remove the StorageDead statements of nthpat and its projections
            let mut new_blocks = Vec::new();
            for (block_id, block) in body.blocks().iter().enumerate() {
                let mut new_stmts = Vec::new();
                for stmt in block.statements.iter() {
                    match &stmt.kind {
                        StatementKind::StorageDead(local) => {
                            if !firstprj_nthprj.values().contains(local) {
                                new_stmts.push(stmt.clone())
                            }
                        }
                        StatementKind::StorageLive(local) => {
                            if !(firstprj_nthprj.values().contains(local)
                                && block_id == nth_blockid + 1)
                            {
                                new_stmts.push(stmt.clone())
                            }
                        }
                        _ => new_stmts.push(stmt.clone()),
                    }
                }
                new_blocks.push((block_id, new_stmts));
            }

            for (block_id, stmts) in new_blocks {
                body.replace_statements(
                    &SourceInstruction::Statement { idx: 0, bb: block_id },
                    stmts,
                );
            }
        }
    }

    fn is_loop_head(&self, body: &MutableBody, tcx: TyCtxt, block_idx: usize) -> bool {
        let terminator = body.blocks()[block_idx].terminator.clone();
        if let TerminatorKind::Call {
            func: terminator_func,
            args: terminator_args,
            destination: _,
            target: _,
            unwind: _,
        } = &terminator.kind
        {
            // Get the function signature of the terminator call.
            let Some(RigidTy::FnDef(fn_def, _)) = terminator_func
                .ty(body.locals())
                .ok()
                .map(|fn_ty| fn_ty.kind().rigid().unwrap().clone())
            else {
                return false;
            };
            // The basic blocks end with register functions are loop head blocks.
            KaniAttributes::for_def_id(tcx, fn_def.def_id()).fn_marker()
                == Some(Symbol::intern("kani_register_loop_contract"))
                && matches!(&terminator_args[1], Operand::Constant(op) if op.const_.eval_target_usize().unwrap() == 0)
        } else {
            false
        }
    }

    //Get all loop-positions: (loop_head_id, loop_latch_id) in the body
    fn get_loop_positions(&self, body: &MutableBody, tcx: TyCtxt) -> Vec<(usize, usize)> {
        let mut loop_pos: Vec<(usize, usize)> = Vec::new();
        for (block_idx, _) in body.blocks().iter().enumerate() {
            if self.is_loop_head(body, tcx, block_idx) {
                let loop_latch_id = self.get_last_loop_latch_id(body, block_idx);
                loop_pos.push((block_idx, loop_latch_id));
            }
        }
        loop_pos
    }

    //Get the associated loop-head of a block_id
    fn get_associated_loop_head(
        &self,
        block_idx: usize,
        loop_positions: &Vec<(usize, usize)>,
    ) -> Option<usize> {
        let mut current_loop_head: Option<usize> = None;
        for (loop_head_idx, loop_latch_idx) in loop_positions {
            if block_idx > *loop_head_idx && block_idx <= *loop_latch_idx {
                current_loop_head = Some(*loop_head_idx);
            }
        }
        current_loop_head
    }

    //Create a Hashmap for a block_id and its associated loop-head
    fn get_associated_loop_head_hashmap(
        &self,
        body: &MutableBody,
        tcx: TyCtxt,
    ) -> HashMap<usize, usize> {
        let loop_positions = self.get_loop_positions(body, tcx);
        let mut loop_head_map: HashMap<usize, usize> = HashMap::new();
        for (block_idx, _) in body.blocks().iter().enumerate() {
            let loop_head = self.get_associated_loop_head(block_idx, &loop_positions);
            if let Some(loop_head) = loop_head {
                loop_head_map.insert(block_idx, loop_head);
            }
        }
        loop_head_map
    }

    ///In case of nested loop, if a variable is declared and initiated inside a loop body, and assigned inside an inner-loop,
    ///then CBMC cannot infer the assign clause for the inner-loop after the loop-contract transformation.
    //Move all variables initiation using assign inside the loop body to the loop-head
    fn move_storagelive_assign_to_loophead(
        &self,
        body: &mut MutableBody,
        loop_head_map: &HashMap<usize, usize>,
    ) {
        let mut add_assign_list: Vec<(usize, Statement)> = Vec::new();
        let mut found_local_list: Vec<usize> = Vec::new();
        let localvars = self.get_user_defined_variables(body);
        let mut blocks_stmts: Vec<(usize, Vec<Statement>)> = Vec::new();
        for (block_idx, block) in body.blocks().iter().enumerate() {
            if loop_head_map.get(&block_idx).is_none() {
                blocks_stmts.push((block_idx, block.statements.clone()));
                continue;
            }
            let closest_loop_head = *loop_head_map.get(&block_idx).unwrap();
            let stmts_len = block.statements.len();
            let mut new_stmts: Vec<Statement> = Vec::new();
            let mut stmt_idx = 0;
            while stmt_idx < stmts_len {
                let stmt = block.statements[stmt_idx].clone();
                if stmt_idx + 1 >= stmts_len {
                    new_stmts.push(stmt.clone());
                    break;
                }
                match stmt.kind {
                    StatementKind::StorageLive(local)
                        if (localvars.contains(&local) && !found_local_list.contains(&local)) =>
                    {
                        let next_stmt = block.statements[stmt_idx + 1].clone();
                        //Case 1: StorageLive followed by an assign
                        if matches!(next_stmt.kind.clone(), StatementKind::Assign(lhs,_) if lhs.local == local)
                        {
                            found_local_list.push(local);
                            add_assign_list.push((closest_loop_head, stmt.clone()));
                            add_assign_list.push((closest_loop_head, next_stmt.clone()));
                            new_stmts.push(next_stmt.clone());
                            stmt_idx += 2;
                            continue;
                        }
                        //Case 2: for Clone(): StorageLive followed by an StorageLive of a temp var, an assign ref of the temp var,
                        //Then an assign of the current local, then a StorageDead of the temp var
                        if let StatementKind::StorageLive(temp_local) = next_stmt.kind.clone()
                            && let Some(third_stmt) = block.statements.get(stmt_idx + 2)
                            && let Some(fourth_stmt) = block.statements.get(stmt_idx + 3)
                            && let Some(fifth_stmt) = block.statements.get(stmt_idx + 4)
                            && matches!(third_stmt.kind.clone(), StatementKind::Assign(lhs, _) if lhs.local == temp_local)
                            && matches!(fourth_stmt.kind.clone(), StatementKind::Assign(lhs, _) if lhs.local == local)
                            && matches!(fifth_stmt.kind.clone(), StatementKind::StorageDead(dead_local) if dead_local == temp_local)
                        {
                            found_local_list.push(local);
                            add_assign_list.push((closest_loop_head, stmt.clone()));
                            add_assign_list.push((closest_loop_head, next_stmt.clone()));
                            add_assign_list.push((closest_loop_head, third_stmt.clone()));
                            add_assign_list.push((closest_loop_head, fourth_stmt.clone()));
                            add_assign_list.push((closest_loop_head, fifth_stmt.clone()));
                            new_stmts.push(next_stmt.clone());
                            new_stmts.push(third_stmt.clone());
                            new_stmts.push(fourth_stmt.clone());
                            new_stmts.push(fifth_stmt.clone());
                            stmt_idx += 5;
                            continue;
                        }
                    }
                    _ => (),
                }
                new_stmts.push(stmt.clone());
                stmt_idx += 1;
            }
            blocks_stmts.push((block_idx, new_stmts));
        }

        for (block_idx, new_stmts) in blocks_stmts {
            body.replace_statements(&SourceInstruction::Terminator { bb: block_idx }, new_stmts);
        }

        for (block_idx, stmt) in add_assign_list {
            body.insert_stmt(
                stmt,
                &mut SourceInstruction::Terminator { bb: block_idx },
                InsertPosition::Before,
            );
        }
    }

    /// Add `StorageLive(x); x = copy x` at the head of each loop with a contract, for each
    /// variable `x` that a loop nested in it may write and that the loop also reads or writes
    /// outside that nested loop. The nested loop may write `x` directly, or through a pointer if
    /// the enclosing loop borrows `x` mutably.
    ///
    /// MIR declares all locals at function scope, and CBMC treats a variable that no instruction
    /// outside a loop reads or writes as local to that loop, leaving it out of the loop's write
    /// set. A nested loop that writes such a variable does not treat it as local, because the
    /// enclosing loop body uses it, so the nested loop's write set is not included in the
    /// enclosing one (#4159). A write before the enclosing loop makes the variable non-local
    /// there too, whatever the value written. The calls that initialize the variable in the loop
    /// body must not be executed at the head: they would run before the loop guard, even when
    /// the loop does not run (#4982).
    ///
    /// The variable is then havocked at the loop head. `transform_bb` copies the `StorageLive` to
    /// the new loop latch, so that the variable is live again there if the body ended its
    /// storage, and the havoc does not write a dead object.
    fn write_nested_loop_variables_at_heads(
        &self,
        body: &mut MutableBody,
        loop_head_map: &HashMap<usize, usize>,
    ) {
        // The loops that contain a block, innermost first.
        let enclosing_loops = |block: usize| {
            std::iter::successors(loop_head_map.get(&block).copied(), |head| {
                loop_head_map.get(head).copied()
            })
            .collect::<Vec<_>>()
        };
        let variables = self.get_storage_moving_variables(body);
        let mut touch_list: Vec<(usize, usize)> = Vec::new();
        // Each loop head that is in the body of another loop is a nested loop.
        for (&inner_head, &outer_head) in loop_head_map {
            if !loop_head_map.values().any(|head| *head == inner_head) {
                continue;
            }
            let mut written_inside: HashSet<usize> = HashSet::new();
            let mut accessed_outside: HashSet<usize> = HashSet::new();
            let mut borrowed_mutably_outside: HashSet<usize> = HashSet::new();
            for &block_idx in loop_head_map.keys() {
                let loops = enclosing_loops(block_idx);
                let block = &body.blocks()[block_idx];
                if loops.contains(&inner_head) {
                    written_inside
                        .extend(block.statements.iter().flat_map(locals_written_by_statement));
                    written_inside.extend(locals_written_by_terminator(&block.terminator));
                } else if loops.contains(&outer_head) {
                    accessed_outside
                        .extend(block.statements.iter().flat_map(locals_accessed_by_statement));
                    accessed_outside.extend(locals_accessed_by_terminator(&block.terminator));
                    borrowed_mutably_outside
                        .extend(block.statements.iter().filter_map(local_borrowed_mutably));
                }
            }
            for &local in &variables {
                if (written_inside.contains(&local) || borrowed_mutably_outside.contains(&local))
                    && accessed_outside.contains(&local)
                    && !touch_list.contains(&(outer_head, local))
                {
                    touch_list.push((outer_head, local));
                }
            }
        }
        touch_list.sort();
        for (loop_head, local) in touch_list {
            let source_info =
                synthetic_source_info(body.blocks()[loop_head].terminator.source_info.span);
            let storage_live = Statement {
                kind: StatementKind::StorageLive(local),
                source_info: source_info.clone(),
            };
            let touch = Statement {
                kind: StatementKind::Assign(
                    Place::from(local),
                    Rvalue::Use(Operand::Copy(Place::from(local)), WithRetag::No),
                ),
                source_info,
            };
            for stmt in [storage_live, touch] {
                body.insert_stmt(
                    stmt,
                    &mut SourceInstruction::Terminator { bb: loop_head },
                    InsertPosition::Before,
                );
            }
        }
    }

    //Move all storagedead inside the loop body to the loop termination block
    fn move_storagedead(&self, body: &mut MutableBody, src_block_idx: usize, dst_block_idx: usize) {
        let localvars = self.get_user_defined_variables(body);
        let storagedead_stmts: Vec<_> = body.blocks()[src_block_idx]
            .clone()
            .statements
            .iter()
            .filter(
                |stmt| matches!(stmt.kind, StatementKind::StorageDead(x) if localvars.contains(&x)),
            )
            .cloned()
            .collect();
        let other_stmts: Vec<_> = body.blocks()[src_block_idx]
            .clone()
            .statements
            .iter()
            .filter(|stmt| !matches!(stmt.kind, StatementKind::StorageDead(x) if localvars.contains(&x)))
            .cloned()
            .collect();
        body.replace_statements(&SourceInstruction::Terminator { bb: src_block_idx }, other_stmts);
        let mut new_stmts = body.blocks()[dst_block_idx].statements.clone();
        let dst_block_stmt_kind: Vec<_> = new_stmts.iter().map(|st| st.kind.clone()).collect();
        for stmt in storagedead_stmts.iter() {
            if !dst_block_stmt_kind.contains(&stmt.kind) {
                new_stmts.push(stmt.clone())
            }
        }
        body.replace_statements(&SourceInstruction::Terminator { bb: dst_block_idx }, new_stmts);
    }

    //Get the associated final loop-latch-id of a loop-head-id
    fn get_last_loop_latch_id(&self, body: &MutableBody, loop_head_id: usize) -> usize {
        let mut loop_latch_id = loop_head_id;
        for (bb_idx, block) in body.blocks().iter().enumerate() {
            match block.terminator.kind {
                TerminatorKind::Goto { target }
                    if (target == loop_head_id && bb_idx > loop_head_id) =>
                {
                    loop_latch_id = bb_idx;
                }
                _ => (),
            }
        }
        loop_latch_id
    }

    //Get the all associated loop-latch-ids of a loop-head-id
    fn get_all_loop_latch_ids(&self, body: &MutableBody, loop_head_id: usize) -> Vec<usize> {
        let mut loop_latch_ids = Vec::new();
        for (bb_idx, block) in body.blocks().iter().enumerate() {
            match block.terminator.kind {
                TerminatorKind::Goto { target }
                    if (target == loop_head_id && bb_idx > loop_head_id) =>
                {
                    loop_latch_ids.push(bb_idx);
                }
                _ => (),
            }
        }
        loop_latch_ids
    }

    /// We only support closure arguments that are either `copy`` or `move`` of reference of user variables.
    fn is_supported_argument_of_closure(&self, rv: &Rvalue, body: &MutableBody) -> bool {
        let var_debug_info = &body.var_debug_info();
        matches!(rv, Rvalue::Ref(_, _, place) if
        var_debug_info.iter().any(|info|
            matches!(&info.value, VarDebugInfoContents::Place(debug_place) if *place == *debug_place)
        ))
    }

    /// This function transform the function body as described in fn transform.
    /// It is the core of fn transform, and is separated just to avoid code repetition.
    fn transform_body_with_loop(&mut self, tcx: TyCtxt, body: Body) -> (bool, Body) {
        let mut new_body = MutableBody::from(body);
        self.replace_first_pat_by_nth_pat(&mut new_body);
        let loop_head_map = self.get_associated_loop_head_hashmap(&new_body, tcx);
        self.move_storagelive_assign_to_loophead(&mut new_body, &loop_head_map);
        self.write_nested_loop_variables_at_heads(&mut new_body, &loop_head_map);
        let mut contain_loop_contracts: bool = false;

        // Visit basic blocks in control flow order (BFS).
        let mut visited: HashSet<BasicBlockIdx> = HashSet::new();
        let mut queue: VecDeque<BasicBlockIdx> = VecDeque::new();
        // Visit blocks in loops only when there is no blocks in queue.
        let mut loop_queue: VecDeque<BasicBlockIdx> = VecDeque::new();
        queue.push_back(0);

        while let Some(bb_idx) = queue.pop_front().or_else(|| loop_queue.pop_front()) {
            visited.insert(bb_idx);

            let terminator = new_body.blocks()[bb_idx].terminator.clone();

            let is_loop_head = self.transform_bb(tcx, &mut new_body, bb_idx);
            contain_loop_contracts |= is_loop_head;

            // Add successors of the current basic blocks to
            // the visiting queue.
            for to_visit in terminator.successors() {
                if !visited.contains(&to_visit) {
                    if is_loop_head {
                        loop_queue.push_back(to_visit);
                    } else {
                        queue.push_back(to_visit)
                    };
                }
            }
        }
        (contain_loop_contracts, new_body.into())
    }

    /// Transform loops with contracts from
    ///    ```ignore
    ///    bb_idx: {
    ///         loop_head_stmts
    ///         _v = kani_register_loop_contract(move args) -> [return: terminator_target];
    ///    }
    ///
    ///    ...
    ///    loop_body_blocks
    ///    ...
    ///
    ///    loop_latch_block: {
    ///         loop_latch_stmts
    ///         goto -> bb_idx;
    ///    }
    ///    ```
    ///    to blocks
    ///    ```ignore
    ///    bb_idx: {
    ///         loop_head_stmts
    ///         _v = true
    ///         goto -> terminator_target
    ///    }
    ///
    ///    ...
    ///    loop_body_blocks
    ///    ...
    ///
    ///    loop_latch_block: {
    ///         loop_latch_stmts
    ///         goto -> bb_new_loop_latch;
    ///    }
    ///
    ///    bb_new_loop_latch: {
    ///         loop_head_body
    ///         _v = kani_register_loop_contract(move args) -> [return: terminator_target];
    ///    }
    ///    ```
    fn transform_bb(&mut self, tcx: TyCtxt, new_body: &mut MutableBody, bb_idx: usize) -> bool {
        let terminator = new_body.blocks()[bb_idx].terminator.clone();
        let mut contain_loop_contracts = false;

        // Redirect loop latches to the new latches.
        if let TerminatorKind::Goto { target: terminator_target } = &terminator.kind
            && self.new_loop_latches.contains_key(terminator_target)
        {
            new_body.replace_terminator(
                &SourceInstruction::Terminator { bb: bb_idx },
                Terminator {
                    kind: TerminatorKind::Goto { target: self.new_loop_latches[terminator_target] },
                    source_info: synthetic_source_info(terminator.source_info.span),
                },
            );
        }

        if let TerminatorKind::SwitchInt { discr, targets } = &terminator.kind {
            let new_branches: Vec<_> = targets
                .branches()
                .map(|(a, b)| {
                    if self.new_loop_latches.contains_key(&b) {
                        (a, self.new_loop_latches[&b])
                    } else {
                        (a, b)
                    }
                })
                .collect();

            let new_otherwise = if self.new_loop_latches.contains_key(&targets.otherwise()) {
                self.new_loop_latches[&targets.otherwise()]
            } else {
                targets.otherwise()
            };

            let new_targets = SwitchTargets::new(new_branches, new_otherwise);
            new_body.replace_terminator(
                &SourceInstruction::Terminator { bb: bb_idx },
                Terminator {
                    kind: TerminatorKind::SwitchInt { discr: discr.clone(), targets: new_targets },
                    source_info: synthetic_source_info(terminator.source_info.span),
                },
            );
        }

        // Transform loop heads with loop contracts.
        if let TerminatorKind::Call {
            func: terminator_func,
            args: terminator_args,
            destination: terminator_destination,
            target: terminator_target,
            unwind: terminator_unwind,
        } = &terminator.kind
        {
            // Get the function signature of the terminator call.
            let Some(RigidTy::FnDef(fn_def, genarg)) = terminator_func
                .ty(new_body.locals())
                .ok()
                .map(|fn_ty| fn_ty.kind().rigid().unwrap().clone())
            else {
                return false;
            };

            // The basic blocks end with register functions are loop head blocks.
            if KaniAttributes::for_def_id(tcx, fn_def.def_id()).fn_marker()
                == Some(Symbol::intern("kani_register_loop_contract"))
                && matches!(&terminator_args[1], Operand::Constant(op) if op.const_.eval_target_usize().unwrap() == 0)
            {
                let loop_termination_block_id = *new_body.blocks()[terminator_target.unwrap()]
                    .terminator
                    .clone()
                    .successors()
                    .first()
                    .unwrap();
                let loop_latch_ids = self.get_all_loop_latch_ids(new_body, bb_idx);
                for loop_latch_id in loop_latch_ids {
                    self.move_storagedead(new_body, loop_latch_id, loop_termination_block_id);
                }

                // Check if the MIR satisfy the assumptions of this transformation.
                if !new_body.blocks()[terminator_target.unwrap()].statements.is_empty()
                    || !matches!(
                        new_body.blocks()[terminator_target.unwrap()].terminator.kind,
                        TerminatorKind::SwitchInt { .. }
                    )
                {
                    unreachable!(
                        "The assumptions for loop-contracts transformation are violated by some other transformation. \
                    Please report github.com/model-checking/kani/issues/new?template=bug_report.md"
                    );
                }
                let GenericArgKind::Type(arg_ty) = genarg.0[0] else { return false };
                let TyKind::RigidTy(RigidTy::Closure(_, genarg)) = arg_ty.kind() else {
                    return false;
                };
                // We look for the args' types of the kani_register_loop_contract function
                // They are always stored in a tuple, which is next to the FnPtr generic args of kani_registered_loop_contract fn
                // All the generic args before the FnPtr are from the outer function
                let mut fnptrpos = 0;
                for (i, arg) in genarg.0.iter().enumerate() {
                    if let GenericArgKind::Type(arg_ty) = arg
                        && let TyKind::RigidTy(RigidTy::FnPtr(_)) = arg_ty.kind()
                    {
                        fnptrpos = i;
                        break;
                    }
                }
                let GenericArgKind::Type(arg_ty) = genarg.0[fnptrpos + 1] else { return false };
                let TyKind::RigidTy(RigidTy::Tuple(args)) = arg_ty.kind() else { return false };
                // Check if the invariant involves any local variable
                if !args.is_empty() {
                    let ori_condition_bb_idx =
                        new_body.blocks()[terminator_target.unwrap()].terminator.successors()[1];
                    self.make_invariant_closure_alive(new_body, ori_condition_bb_idx);
                }

                contain_loop_contracts = true;

                // Collect supported vars assigned in the block.
                // And check if all arguments of the closure is supported.
                let mut supported_vars: Vec<usize> = Vec::new();
                // All user variables are support
                supported_vars.extend(new_body.var_debug_info().iter().filter_map(|info| {
                    match &info.value {
                        VarDebugInfoContents::Place(debug_place) => Some(debug_place.local),
                        _ => None,
                    }
                }));

                // For each assignment in the loop head block,
                // if it assigns to the closure place, we check if all arguments are supported;
                // if it assigns to other places, we cache if the assigned places are supported.
                for stmt in &new_body.blocks()[bb_idx].statements {
                    if let StatementKind::Assign(place, rvalue) = &stmt.kind {
                        match rvalue {
                            Rvalue::Ref(_,_,rplace) | Rvalue::CopyForDeref(rplace) | Rvalue::Use(Operand::Copy(rplace), _) => {
                                if supported_vars.contains(&rplace.local) {
                                    supported_vars.push(place.local);
                                } }
                            Rvalue::Aggregate(AggregateKind::Closure(..), closure_args) => {
                                if closure_args.iter().any(|arg| !matches!(arg, Operand::Copy(arg_place) | Operand::Move(arg_place) if supported_vars.contains(&arg_place.local))) {
                                    unreachable!(
                                            "The loop invariant support only reference of user variables. The provided invariants contain unsupported dereference. \
                                            Please report github.com/model-checking/kani/issues/new?template=bug_report.md"
                                        );
                                }
                            }
                            _ => {
                                if self.is_supported_argument_of_closure(rvalue, new_body) {
                                    supported_vars.push(place.local);
                                }
                            }
                        }
                    }
                }

                // Replace the original loop head block
                // ```ignore
                // bb_idx: {
                //          loop_head_stmts
                //          _v = kani_register_loop_contract(move args) -> [return: terminator_target];
                // }
                // ```
                // with
                // ```ignore
                // bb_idx: {
                //          loop_head_stmts
                //          _v = true;
                //          goto -> terminator_target
                // }
                // ```
                new_body.assign_to(
                    terminator_destination.clone(),
                    Rvalue::Use(
                        Operand::Constant(ConstOperand {
                            span: terminator.source_info.span,
                            user_ty: None,
                            const_: MirConst::from_bool(true),
                        }),
                        WithRetag::No,
                    ),
                    &mut SourceInstruction::Terminator { bb: bb_idx },
                    InsertPosition::Before,
                );
                let new_latch_block = self.get_loop_head_block(&new_body.blocks()[bb_idx]);

                // Insert a new basic block as the loop latch block, and later redirect
                // all latches to the new loop latch block.
                // -----
                // bb_new_loop_latch: {
                //    _v = kani_register_loop_contract(move args) -> [return: terminator_target];
                // }
                new_body.insert_bb(
                    new_latch_block,
                    &mut SourceInstruction::Terminator { bb: bb_idx },
                    InsertPosition::After,
                );
                // Update the argument `transformed` to 1 to avoid double transformation.
                let new_args = vec![
                    terminator_args[0].clone(),
                    Operand::Constant(ConstOperand {
                        span: terminator.source_info.span,
                        user_ty: None,
                        const_: MirConst::try_from_uint(1, UintTy::Usize).unwrap(),
                    }),
                ];
                new_body.replace_terminator(
                    &SourceInstruction::Terminator { bb: new_body.blocks().len() - 1 },
                    Terminator {
                        kind: TerminatorKind::Call {
                            func: terminator_func.clone(),
                            args: new_args,
                            destination: terminator_destination.clone(),
                            target: *terminator_target,
                            unwind: *terminator_unwind,
                        },
                        source_info: synthetic_source_info(terminator.source_info.span),
                    },
                );
                new_body.replace_terminator(
                    &SourceInstruction::Terminator { bb: bb_idx },
                    Terminator {
                        kind: TerminatorKind::Goto { target: terminator_target.unwrap() },
                        source_info: synthetic_source_info(terminator.source_info.span),
                    },
                );
                // Cache the new loop latch.
                self.new_loop_latches.insert(bb_idx, new_body.blocks().len() - 1);
            }
        }
        contain_loop_contracts
    }
}

/// The local of `place` if writing `place` writes that local (the place has no dereference).
fn written_local(place: &Place) -> Option<usize> {
    (!place.projection.iter().any(|elem| matches!(elem, ProjectionElem::Deref)))
        .then_some(place.local)
}

/// The local of a place that a statement borrows mutably (`&mut`, `&raw mut` or a mutable
/// reborrow).
fn local_borrowed_mutably(stmt: &Statement) -> Option<usize> {
    match &stmt.kind {
        StatementKind::Assign(
            _,
            Rvalue::Ref(_, BorrowKind::Mut { .. }, place)
            | Rvalue::AddressOf(RawPtrKind::Mut, place)
            | Rvalue::Reborrow(_, Mutability::Mut, place),
        ) => Some(place.local),
        _ => None,
    }
}

/// The locals that a statement may write: the local of an assigned place, and the local of a
/// place borrowed mutably (the borrow may be used for a write later).
fn locals_written_by_statement(stmt: &Statement) -> Vec<usize> {
    match &stmt.kind {
        StatementKind::Assign(place, _) => {
            written_local(place).into_iter().chain(local_borrowed_mutably(stmt)).collect()
        }
        StatementKind::SetDiscriminant { place, .. } => written_local(place).into_iter().collect(),
        _ => vec![],
    }
}

/// The locals that a terminator may write: the destination of a call and a dropped place.
fn locals_written_by_terminator(terminator: &Terminator) -> Vec<usize> {
    match &terminator.kind {
        TerminatorKind::Call { destination, .. } => {
            written_local(destination).into_iter().collect()
        }
        TerminatorKind::Drop { place, .. } => vec![place.local],
        _ => vec![],
    }
}

/// The locals that a place expression reads or writes: the locals of its index projections and
/// its base local, except when the place is only borrowed without a dereference (as for CBMC,
/// taking the address of a variable does not access it).
fn locals_accessed_by_place(place: &Place, borrowed: bool) -> Vec<usize> {
    let mut accessed: Vec<usize> = place
        .projection
        .iter()
        .filter_map(
            |elem| if let ProjectionElem::Index(local) = elem { Some(*local) } else { None },
        )
        .collect();
    if !borrowed || written_local(place).is_none() {
        accessed.push(place.local);
    }
    accessed
}

fn locals_accessed_by_operand(operand: &Operand) -> Vec<usize> {
    match operand {
        Operand::Constant(_) | Operand::RuntimeChecks(_) => vec![],
        Operand::Copy(place) | Operand::Move(place) => locals_accessed_by_place(place, false),
    }
}

fn locals_accessed_by_rvalue(rvalue: &Rvalue) -> Vec<usize> {
    match rvalue {
        Rvalue::AddressOf(_, place) | Rvalue::Reborrow(_, _, place) | Rvalue::Ref(_, _, place) => {
            locals_accessed_by_place(place, true)
        }
        Rvalue::Aggregate(_, operands) => {
            operands.iter().flat_map(locals_accessed_by_operand).collect()
        }
        Rvalue::BinaryOp(_, lhs, rhs) | Rvalue::CheckedBinaryOp(_, lhs, rhs) => {
            [lhs, rhs].into_iter().flat_map(locals_accessed_by_operand).collect()
        }
        Rvalue::Cast(_, operand, _)
        | Rvalue::Repeat(operand, _)
        | Rvalue::UnaryOp(_, operand)
        | Rvalue::Use(operand, _) => locals_accessed_by_operand(operand),
        Rvalue::CopyForDeref(place) | Rvalue::Discriminant(place) | Rvalue::Len(place) => {
            locals_accessed_by_place(place, false)
        }
        Rvalue::ThreadLocalRef(_) => vec![],
    }
}

/// The locals that a statement reads or writes. Storage markers are not accesses.
fn locals_accessed_by_statement(stmt: &Statement) -> Vec<usize> {
    match &stmt.kind {
        StatementKind::Assign(place, rvalue) => {
            let mut accessed = locals_accessed_by_rvalue(rvalue);
            accessed.extend(locals_accessed_by_place(place, false));
            accessed
        }
        StatementKind::Intrinsic(NonDivergingIntrinsic::Assume(operand)) => {
            locals_accessed_by_operand(operand)
        }
        StatementKind::Intrinsic(NonDivergingIntrinsic::CopyNonOverlapping(copy)) => {
            [&copy.src, &copy.dst, &copy.count]
                .into_iter()
                .flat_map(locals_accessed_by_operand)
                .collect()
        }
        StatementKind::SetDiscriminant { place, .. } => locals_accessed_by_place(place, false),
        _ => vec![],
    }
}

/// The locals that a terminator reads or writes.
fn locals_accessed_by_terminator(terminator: &Terminator) -> Vec<usize> {
    match &terminator.kind {
        TerminatorKind::Assert { cond, .. } => locals_accessed_by_operand(cond),
        TerminatorKind::Call { func, args, destination, .. } => {
            let mut accessed: Vec<usize> =
                args.iter().chain([func]).flat_map(locals_accessed_by_operand).collect();
            accessed.extend(locals_accessed_by_place(destination, false));
            accessed
        }
        TerminatorKind::Drop { place, .. } => locals_accessed_by_place(place, false),
        TerminatorKind::SwitchInt { discr, .. } => locals_accessed_by_operand(discr),
        _ => vec![],
    }
}
