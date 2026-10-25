"""Constrained counterfactual optimization service with DiCE and strict post-generation feasibility validation."""

import time
from typing import Any, Literal
import numpy as np
import pandas as pd
import dice_ml

from backend.app.schemas.counterfactual import (
    CounterfactualCandidate,
    CounterfactualMetrics,
    CounterfactualResult,
    DerivedRecalculation,
    FeatureChange,
)
from backend.app.schemas.feature_schema import FeatureSchema
from backend.app.schemas.prediction import InputValidationError
from backend.app.services.prediction import PredictionService


class CounterfactualService:
    """Generates and validates actionable, constrained counterfactual remediation scenarios."""

    EXACT_NO_FEASIBLE_MESSAGE = "No feasible counterfactual found under configured constraints"

    def __init__(
        self,
        prediction_service: PredictionService,
        feature_schema: FeatureSchema,
        training_df: pd.DataFrame,
        dice_method: Literal["genetic", "random", "kdtree"] = "genetic",
        target_threshold: float = 0.35,
    ):
        self.prediction_service = prediction_service
        self.feature_schema = feature_schema
        self.training_df = training_df.copy()
        self.dice_method = dice_method
        self.target_threshold = target_threshold

        # Define actionable primitive features eligible for perturbation
        actionable_defs = self.feature_schema.get_actionable_features()
        self.features_to_vary = [f.name for f in actionable_defs if not f.derived]

        # Define immutable features that must NEVER change
        immutable_defs = self.feature_schema.get_immutable_features()
        self.immutable_features = [f.name for f in immutable_defs]

        # Initialize DiCE data and model objects
        self._init_dice_engine()

    def _init_dice_engine(self) -> None:
        """Initialize Microsoft's DiCE engine."""
        # Ensure training DataFrame has the binary target column
        if "target" not in self.training_df.columns:
            # Generate binary target based on model probability for background points
            probs = []
            for _, row in self.training_df.iterrows():
                try:
                    res = self.prediction_service.predict(row.to_dict())
                    probs.append(1 if res.default_probability >= self.target_threshold else 0)
                except Exception:
                    probs.append(0)
            self.training_df["target"] = probs

        canonical_cols = [c for c in self.feature_schema.feature_order if c in self.training_df.columns]
        dice_df = self.training_df[canonical_cols + ["target"]].copy()

        self.dice_data = dice_ml.Data(
            dataframe=dice_df,
            continuous_features=canonical_cols,
            outcome_name="target",
        )

        # Scikit-learn compatible wrapper for DiCE
        class SklearnCompatibleWrapper:
            def __init__(self, pred_svc):
                self.pred_svc = pred_svc

            def predict(self, X):
                probs = self.predict_proba(X)
                return (probs[:, 1] >= 0.5).astype(int)

            def predict_proba(self, X):
                cols = self.pred_svc.feature_schema.feature_order
                if isinstance(X, pd.DataFrame):
                    df_in = X[cols].copy()
                elif isinstance(X, (list, np.ndarray)):
                    df_in = pd.DataFrame(X, columns=cols)
                else:
                    df_in = pd.DataFrame([X])[cols]

                import xgboost as xgb_inner
                dmat = xgb_inner.DMatrix(df_in[cols])
                raw_probs = self.pred_svc.model.predict(dmat)
                out = []
                for p in raw_probs:
                    p_val = float(p)
                    out.append([1.0 - p_val, p_val])
                return np.array(out)

        self.dice_model_wrapper = SklearnCompatibleWrapper(self.prediction_service)
        self.dice_model = dice_ml.Model(model=self.dice_model_wrapper, backend="sklearn")
        try:
            self.dice_explainer = dice_ml.Dice(
                self.dice_data,
                self.dice_model,
                method=self.dice_method,
            )
        except Exception:
            # Fallback to random if genetic setup has issues with local dataset
            self.dice_explainer = dice_ml.Dice(
                self.dice_data,
                self.dice_model,
                method="random",
            )

    def validate_candidate(
        self,
        candidate_id: str,
        original_dict: dict[str, float],
        candidate_dict: dict[str, float],
    ) -> CounterfactualCandidate:
        """Rigorous post-generation validation layer.

        Enforces:
        1. Hard immutability checks on frozen / sensitive features.
        2. Recalculation of derived ratios from updated primitives.
        3. Feature range / domain bounds.
        4. Exact re-scoring via production inference pipeline.
        """
        changes: list[FeatureChange] = []
        derived_recalculations: list[DerivedRecalculation] = []
        constraints_passed: list[str] = []
        constraints_failed: list[str] = []

        # 1. Check immutability
        for imm in self.immutable_features:
            orig_val = original_dict.get(imm)
            cand_val = candidate_dict.get(imm)
            if orig_val is not None and cand_val is not None:
                diff = abs(cand_val - orig_val)
                if diff > 1e-4:
                    constraints_failed.append(
                        f"Immutable feature '{imm}' was altered: original={orig_val}, proposed={cand_val} (diff={diff:.4f})"
                    )
                else:
                    constraints_passed.append(f"Immutable feature '{imm}' preserved")

        # 2. Record primitive changes on actionable levers
        for act in self.features_to_vary:
            orig_val = original_dict.get(act, 0.0)
            cand_val = candidate_dict.get(act, orig_val)
            delta = cand_val - orig_val
            if abs(delta) > 1e-4:
                changes.append(
                    FeatureChange(
                        feature=act,
                        current=round(orig_val, 4),
                        proposed=round(cand_val, 4),
                        delta=round(delta, 4),
                    )
                )

        # 3. Check domain ranges on primitives
        for act in self.features_to_vary:
            feat_def = self.feature_schema.get_feature(act)
            if feat_def and feat_def.allowed_range:
                cand_val = candidate_dict.get(act)
                if cand_val is not None:
                    if feat_def.allowed_range.min_value is not None and cand_val < feat_def.allowed_range.min_value:
                        constraints_failed.append(
                            f"Feature '{act}' value {cand_val} below minimum bound {feat_def.allowed_range.min_value}"
                        )
                    elif feat_def.allowed_range.max_value is not None and cand_val > feat_def.allowed_range.max_value:
                        constraints_failed.append(
                            f"Feature '{act}' value {cand_val} above maximum bound {feat_def.allowed_range.max_value}"
                        )
                    else:
                        constraints_passed.append(f"Domain range verified for '{act}'")

        # 4. Recalculate derived ratios from updated primitives
        primitive_inputs = {k: candidate_dict[k] for k in candidate_dict if k in [f.name for f in self.feature_schema.get_primitive_features()]}
        recomputed_derived = self.prediction_service.compute_derived_features(primitive_inputs)

        for der_name, new_val in recomputed_derived.items():
            feat_def = self.feature_schema.get_feature(der_name)
            prev_val = original_dict.get(der_name, 0.0)
            formula_str = feat_def.formula if feat_def else "recomputed from primitives"
            derived_recalculations.append(
                DerivedRecalculation(
                    feature=der_name,
                    formula=formula_str or "formula",
                    previous_value=round(prev_val, 4),
                    new_value=round(new_val, 4),
                )
            )
            # Update candidate dict with authentic recomputed derived value
            candidate_dict[der_name] = new_val
            constraints_passed.append(f"Derived feature '{der_name}' recomputed from primitives")

        # 5. Exact re-score via production inference pipeline
        clean_candidate_input = {**primitive_inputs, **recomputed_derived}
        try:
            rescore_result = self.prediction_service.predict(clean_candidate_input)
            new_prob = rescore_result.default_probability
            target_met = new_prob < self.target_threshold
            if target_met:
                constraints_passed.append(f"Target risk met: {new_prob:.4f} < {self.target_threshold:.4f}")
            else:
                constraints_failed.append(f"Target risk not met: {new_prob:.4f} >= {self.target_threshold:.4f}")
        except InputValidationError as ive:
            target_met = False
            new_prob = 1.0  # Max risk on invalid input
            for err in ive.errors:
                constraints_failed.append(f"Production pipeline rejected input: {err.field} - {err.issue}")

        # Feasibility classification
        if len(constraints_failed) == 0:
            feasibility_status = "feasible"
        elif target_met and len(constraints_failed) > 0:
            feasibility_status = "partially_feasible"
        else:
            feasibility_status = "infeasible"

        caveats = [
            "Candidate scenario for human review; this is not a guarantee of loan approval.",
            "Any modification to loan amount, collateral, or tenor requires credit committee approval.",
        ]

        unmodelled_dependencies = [
            "Legal title search and independent physical appraisal for additional collateral.",
            "Borrower written consent and updated repayment schedule for tenor extension.",
        ]

        return CounterfactualCandidate(
            id=candidate_id,
            changes=changes,
            derived_recalculations=derived_recalculations,
            predicted_probability=round(new_prob, 6),
            target_met=target_met,
            constraints_passed=constraints_passed,
            constraints_failed=constraints_failed,
            feasibility_status=feasibility_status,
            caveats=caveats,
            unmodelled_dependencies=unmodelled_dependencies,
        )

    def _generate_candidate_profiles(
        self,
        applicant_data: dict[str, Any],
        total_cfs: int,
    ) -> tuple[list[dict[str, float]], bool, bool]:
        """Generate candidate raw profiles using DiCE with fallback to constrained domain search."""
        raw_candidates: list[dict[str, float]] = []
        dice_succeeded = False
        fallback_used = False

        # 1. Try DiCE generation
        try:
            query_df, _ = self.prediction_service.prepare_feature_vector(applicant_data)
            dice_features = [f for f in self.features_to_vary if f in self.dice_data.continuous_feature_names]
            dice_exp = self.dice_explainer.generate_counterfactuals(
                query_df,
                total_CFs=total_cfs,
                desired_class=0,
                features_to_vary=dice_features,
            )
            cf_df = dice_exp.cf_examples_list[0].final_cfs_df
            if cf_df is not None and len(cf_df) > 0:
                for _, row in cf_df.iterrows():
                    cand_row = dict(applicant_data)
                    cand_row.update(row.drop("target", errors="ignore").to_dict())
                    raw_candidates.append(cand_row)
                dice_succeeded = True
        except Exception:
            # DiCE may fail to find CFs or throw UserConfigValidationException
            dice_succeeded = False

        # 2. Constrained domain heuristic grid search if DiCE produced fewer than total_cfs
        if len(raw_candidates) < total_cfs:
            fallback_used = True
            orig_credit = float(applicant_data.get("AMT_CREDIT", 1500000.0))
            orig_annuity = float(applicant_data.get("AMT_ANNUITY", 45000.0))
            orig_goods = float(applicant_data.get("AMT_GOODS_PRICE", 1685393.0))

            # Strategic financial restructuring levers:
            # Option 1: Tenor extension (reducing annuity by 25% and 35%)
            # Option 2: Additional collateral (increasing goods/collateral price by 15% and 25%)
            # Option 3: Joint combination of tenor extension + additional collateral
            adjustments = [
                {"AMT_ANNUITY": orig_annuity * 0.70, "AMT_GOODS_PRICE": orig_goods * 1.20},  # Tenor extension + collateral
                {"AMT_ANNUITY": orig_annuity * 0.65, "AMT_GOODS_PRICE": orig_goods * 1.10},  # Major tenor extension
                {"AMT_CREDIT": orig_credit * 0.85, "AMT_ANNUITY": orig_annuity * 0.80},     # Downpayment / credit reduction
                {"AMT_ANNUITY": orig_annuity * 0.90, "AMT_GOODS_PRICE": orig_goods * 1.05},  # Mild restructuring (may miss target)
            ]

            for adj in adjustments:
                if len(raw_candidates) >= total_cfs:
                    break
                cand_copy = dict(applicant_data)
                for k, v in adj.items():
                    cand_copy[k] = v
                raw_candidates.append(cand_copy)

        return raw_candidates, dice_succeeded, fallback_used

    def generate_counterfactuals(
        self,
        applicant_data: dict[str, Any],
        total_cfs: int = 3,
    ) -> CounterfactualResult:
        """Run constrained counterfactual search, validation layer, and metric aggregation."""
        start_time = time.perf_counter()

        # Extract initial state
        _, original_eval_dict = self.prediction_service.prepare_feature_vector(applicant_data)

        # Generate candidates
        raw_candidates, dice_succeeded, fallback_used = self._generate_candidate_profiles(
            applicant_data, total_cfs=total_cfs
        )

        valid_candidates: list[CounterfactualCandidate] = []
        closest_non_satisfying: list[CounterfactualCandidate] = []

        total_evaluated = len(raw_candidates)
        target_met_count = 0
        actionable_count = 0
        target_misses: list[float] = []

        for idx, raw_cand in enumerate(raw_candidates, start=1):
            cand_id = f"CF-{idx}"
            evaluated_cand = self.validate_candidate(
                candidate_id=cand_id,
                original_dict=original_eval_dict,
                candidate_dict=raw_cand,
            )

            if evaluated_cand.target_met:
                target_met_count += 1
                if evaluated_cand.feasibility_status == "feasible":
                    actionable_count += 1
                    valid_candidates.append(evaluated_cand)
                else:
                    closest_non_satisfying.append(evaluated_cand)
            else:
                miss = evaluated_cand.predicted_probability - self.target_threshold
                target_misses.append(max(0.0, miss))
                closest_non_satisfying.append(evaluated_cand)

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        # Calculate metrics with explicit denominator
        denominator_str = f"Evaluated across {total_evaluated} candidate scenario(s)"
        validity = (target_met_count / total_evaluated) if total_evaluated > 0 else 0.0
        actionability = (actionable_count / total_evaluated) if total_evaluated > 0 else 0.0
        failure_rate = 1.0 - validity
        avg_target_miss = float(np.mean(target_misses)) if target_misses else 0.0

        # Sparsity: average number of modified actionable features
        sparsity_counts = [len(c.changes) for c in valid_candidates]
        mean_sparsity = float(np.mean(sparsity_counts)) if sparsity_counts else 0.0

        # Proximity: normalized L1 distance on actionable features
        proximity_scores = []
        for c in valid_candidates:
            diffs = []
            for ch in c.changes:
                feat_def = self.feature_schema.get_feature(ch.feature)
                if feat_def and feat_def.allowed_range and feat_def.allowed_range.max_value and feat_def.allowed_range.min_value:
                    span = feat_def.allowed_range.max_value - feat_def.allowed_range.min_value
                    diffs.append(abs(ch.delta) / max(span, 1.0))
                else:
                    diffs.append(abs(ch.delta) / max(ch.current, 1.0))
            if diffs:
                proximity_scores.append(float(np.mean(diffs)))
        mean_proximity = float(np.mean(proximity_scores)) if proximity_scores else 0.0

        # Diversity: pairwise distance among valid candidates
        if len(valid_candidates) > 1:
            pair_diffs = []
            for i in range(len(valid_candidates)):
                for j in range(i + 1, len(valid_candidates)):
                    ci = {ch.feature: ch.proposed for ch in valid_candidates[i].changes}
                    cj = {ch.feature: ch.proposed for ch in valid_candidates[j].changes}
                    keys = set(ci.keys()).union(set(cj.keys()))
                    d = sum(abs(ci.get(k, 0.0) - cj.get(k, 0.0)) for k in keys)
                    pair_diffs.append(d)
            mean_diversity = float(np.mean(pair_diffs))
        else:
            mean_diversity = 0.0

        metrics = CounterfactualMetrics(
            validity=round(validity, 4),
            actionability=round(actionability, 4),
            generation_failure_rate=round(failure_rate, 4),
            time_to_generate_ms=round(elapsed_ms, 2),
            number_of_candidates=len(valid_candidates),
            proximity=round(mean_proximity, 4),
            diversity=round(mean_diversity, 4),
            ood_flags=0,
            target_miss_magnitude=round(avg_target_miss, 4),
            denominator=denominator_str,
            dice_succeeded=dice_succeeded,
            fallback_used=fallback_used,
            sparsity=round(mean_sparsity, 4),
        )

        no_feasible_found = len(valid_candidates) == 0
        no_feasible_message = self.EXACT_NO_FEASIBLE_MESSAGE if no_feasible_found else None

        return CounterfactualResult(
            schema_version="1.0.0",
            request_config={
                "desired_class": 0,
                "target_threshold": self.target_threshold,
                "features_to_vary": self.features_to_vary,
                "immutable_features": self.immutable_features,
            },
            candidates=valid_candidates,
            no_feasible_found=no_feasible_found,
            no_feasible_message=no_feasible_message,
            closest_non_satisfying=closest_non_satisfying,
            metrics=metrics,
        )
