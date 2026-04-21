from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gc
import h5py
import numpy as np
import pandas as pd
import psutil
import statsmodels.api as sm
from joblib import Parallel, delayed
from statsmodels.regression.mixed_linear_model import MixedLM
from tqdm import tqdm


@dataclass
class CTSConfig:
    h5_file: str
    ref_dose: float = 10.0
    n_jobs: int = 4
    chunk_size: int = 50


class CTSMetaRegression:
    def __init__(self, config: CTSConfig) -> None:
        self.config = config
        self.h5_path = Path(config.h5_file)
        if not self.h5_path.exists():
            raise FileNotFoundError(f"H5 file not found: {self.h5_path}")
        self.n_jobs = config.n_jobs if config.n_jobs > 0 else psutil.cpu_count(logical=True) or 1

    def load_data(self) -> None:
        with h5py.File(self.h5_path, "r") as handle:
            expr_shape = handle["expr"].shape
            raw_gene_names = handle["gene_names"][:]
            raw_sample_names = handle["sample_names"][:]

            if len(raw_gene_names) == expr_shape[1] and len(raw_sample_names) == expr_shape[0]:
                self.expr_matrix = handle["expr"][:].T
                self.n_genes = expr_shape[1]
                self.n_samples = expr_shape[0]
            else:
                self.expr_matrix = handle["expr"][:]
                self.n_genes = expr_shape[0]
                self.n_samples = expr_shape[1]

        self.gene_names = [self._decode_name(name) for name in raw_gene_names]
        self.sample_names = [self._decode_name(name) for name in raw_sample_names]
        self.sample_info = self._build_sample_info(self.sample_names)

    @staticmethod
    def _decode_name(value) -> str:
        return value.decode() if isinstance(value, bytes) else str(value)

    @staticmethod
    def _build_sample_info(sample_names: List[str]) -> pd.DataFrame:
        rows: List[Dict] = []
        for sample_idx, sample_name in enumerate(sample_names):
            parts = sample_name.split(":")
            if len(parts) < 3:
                continue
            try:
                dose = float(parts[2])
            except ValueError:
                continue
            rows.append(
                {
                    "sample_idx": sample_idx,
                    "sample_id": sample_name,
                    "drug_id": parts[0],
                    "cell_line": parts[1],
                    "dose": dose,
                    "log10_dose": np.log10(dose) if dose > 0 else np.nan,
                }
            )
        return pd.DataFrame(rows).dropna()

    def run(self, output_prefix: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        self.load_data()
        results = self._process_all_genes()
        if not results:
            raise RuntimeError("No CTS results were produced from the input H5 file")

        results_df = pd.DataFrame(results)
        if "method" not in results_df.columns:
            results_df["method"] = "mixed_model"

        consensus_matrix = results_df.pivot(index="drug", columns="gene", values="consensus_z_at_ref_dose").fillna(0)
        heterogeneity_matrix = results_df.pivot(index="drug", columns="gene", values="tau2_heterogeneity").fillna(0)
        method_matrix = results_df.pivot(index="drug", columns="gene", values="method").fillna("no_data")

        prefix = Path(output_prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        consensus_matrix.to_csv(f"{prefix}_consensus.csv")
        heterogeneity_matrix.to_csv(f"{prefix}_heterogeneity.csv")
        results_df.to_csv(f"{prefix}_detailed_results.csv", index=False)
        method_matrix.to_csv(f"{prefix}_method_matrix.csv")
        return consensus_matrix, heterogeneity_matrix, results_df

    def _process_all_genes(self) -> List[Dict]:
        gene_indices = list(range(self.n_genes))
        all_results: List[Dict] = []
        n_chunks = (self.n_genes + self.config.chunk_size - 1) // self.config.chunk_size

        for chunk_idx in range(n_chunks):
            start_idx = chunk_idx * self.config.chunk_size
            end_idx = min(start_idx + self.config.chunk_size, self.n_genes)
            chunk_indices = gene_indices[start_idx:end_idx]
            chunk_results = Parallel(n_jobs=self.n_jobs, backend="loky")(
                delayed(self._process_single_gene)(gene_idx)
                for gene_idx in tqdm(chunk_indices, desc=f"chunk {chunk_idx + 1}/{n_chunks}")
            )
            for gene_result in chunk_results:
                all_results.extend(gene_result)
            if chunk_idx % 5 == 0:
                gc.collect()
        return all_results

    def _process_single_gene(self, gene_idx: int) -> List[Dict]:
        gene_name = self.gene_names[gene_idx]
        gene_expr = self.expr_matrix[gene_idx, :]
        long_rows: List[Dict] = []
        for sample in self.sample_info.itertuples(index=False):
            if sample.sample_idx >= len(gene_expr):
                continue
            long_rows.append(
                {
                    "gene_id": gene_name,
                    "drug_id": sample.drug_id,
                    "cell_line": sample.cell_line,
                    "dose": sample.dose,
                    "log10_dose": sample.log10_dose,
                    "z_score": gene_expr[sample.sample_idx],
                }
            )
        if not long_rows:
            return []

        long_df = pd.DataFrame(long_rows)
        grouped = long_df.groupby(["drug_id", "gene_id", "cell_line", "dose"])
        variance_rows: List[Dict] = []
        for name, group in grouped:
            mean_z = group["z_score"].mean()
            var_z = group["z_score"].var() if len(group) > 1 else 0.001
            n_reps = len(group)
            variance_of_mean = var_z / n_reps if n_reps > 0 else 0.001
            if variance_of_mean > 1e-8 and not np.isnan(mean_z):
                variance_rows.append(
                    {
                        "drug_id": name[0],
                        "gene_id": name[1],
                        "cell_line": name[2],
                        "dose": name[3],
                        "log10_dose": np.log10(name[3]) if name[3] > 0 else np.nan,
                        "mean_z": mean_z,
                        "variance_of_mean_z": variance_of_mean,
                        "n_reps": n_reps,
                    }
                )
        if not variance_rows:
            return []

        variance_df = pd.DataFrame(variance_rows).dropna()
        gene_results: List[Dict] = []
        for (drug_id, gene_id), group_data in variance_df.groupby(["drug_id", "gene_id"]):
            n_obs = len(group_data)
            n_cells = group_data["cell_line"].nunique()
            n_doses = group_data["log10_dose"].nunique()
            result = None
            if n_obs >= 4 and n_cells >= 2 and n_doses >= 2:
                result = self._fit_mixed_model(group_data, drug_id, gene_id)
            elif n_obs >= 4 and n_cells >= 2 and n_doses == 1:
                result = self._fit_intercept_only(group_data, drug_id, gene_id)
            elif n_obs >= 2:
                result = self._fit_weighted_aggregation(group_data, drug_id, gene_id)
            elif n_obs == 1:
                result = self._fit_single_point(group_data, drug_id, gene_id)
            if result is not None:
                gene_results.append(result)
        return gene_results

    def _fit_mixed_model(self, data: pd.DataFrame, drug_id: str, gene_id: str) -> Optional[Dict]:
        try:
            y = data["mean_z"].values
            X = sm.add_constant(data["log10_dose"].values.reshape(-1, 1))
            groups = data["cell_line"].values
            model = MixedLM(y, X, groups=groups)
            result = model.fit(method="lbfgs", maxiter=50, gtol=1e-4, ftol=1e-5, reml=False)
            if not result.converged:
                return None
            pred = result.predict(np.array([[1, np.log10(self.config.ref_dose)]]))[0]
            try:
                pred_se = float(np.sqrt(result.cov_params().iloc[1, 1]))
            except Exception:
                pred_se = 0.0
            try:
                tau2 = float(result.cov_re.iloc[0, 0]) if hasattr(result, "cov_re") else 0.0
            except Exception:
                tau2 = 0.0
            return {
                "drug": drug_id,
                "gene": gene_id,
                "consensus_z_at_ref_dose": pred,
                "se_of_consensus_z": pred_se,
                "tau2_heterogeneity": tau2,
                "n_observations": len(data),
                "converged": bool(result.converged),
                "method": "mixed_model",
            }
        except Exception:
            return None

    def _fit_intercept_only(self, data: pd.DataFrame, drug_id: str, gene_id: str) -> Optional[Dict]:
        try:
            y = data["mean_z"].values
            X = np.ones((len(y), 1))
            groups = data["cell_line"].values
            model = MixedLM(y, X, groups=groups)
            result = model.fit(method="lbfgs", maxiter=50, gtol=1e-4, ftol=1e-5, reml=False)
            if not result.converged:
                return None
            unique_dose = data["dose"].iloc[0]
            ref_log_dose = np.log10(self.config.ref_dose)
            obs_log_dose = np.log10(unique_dose) if unique_dose > 0 else np.nan
            dose_diff = abs(obs_log_dose - ref_log_dose) if not np.isnan(obs_log_dose) else np.inf
            if dose_diff < 0.3:
                dose_penalty = 1.0
                uncertainty_multiplier = 1.0
            elif dose_diff < 1.0:
                dose_penalty = min(1.0, 1.0 / (1.0 + dose_diff * 0.5))
                uncertainty_multiplier = 1.3
            else:
                dose_penalty = min(1.0, 1.0 / (1.0 + dose_diff))
                uncertainty_multiplier = 2.0
            consensus_z = float(result.params[0]) * dose_penalty
            try:
                base_se = float(np.sqrt(result.cov_params().iloc[0, 0]))
            except Exception:
                base_se = 0.0
            extrapolation_uncertainty = dose_diff * 0.1 if dose_diff < np.inf else 1.0
            pred_se = base_se * uncertainty_multiplier + extrapolation_uncertainty
            try:
                tau2 = float(result.cov_re.iloc[0, 0]) if hasattr(result, "cov_re") else 0.0
            except Exception:
                tau2 = 0.0
            return {
                "drug": drug_id,
                "gene": gene_id,
                "consensus_z_at_ref_dose": consensus_z,
                "se_of_consensus_z": pred_se,
                "tau2_heterogeneity": tau2,
                "n_observations": len(data),
                "converged": bool(result.converged),
                "method": "intercept_only_mixed",
            }
        except Exception:
            return None

    def _fit_weighted_aggregation(self, data: pd.DataFrame, drug_id: str, gene_id: str) -> Optional[Dict]:
        try:
            ref_log_dose = np.log10(self.config.ref_dose)
            dose_distances = np.abs(data["log10_dose"].values - ref_log_dose)
            weights = np.exp(-(dose_distances ** 2) / 2)
            weights = weights / np.sum(weights)
            consensus_z = float(np.sum(data["mean_z"].values * weights))
            se_z = float(np.sqrt(np.sum((weights ** 2) * data["variance_of_mean_z"].values)))
            return {
                "drug": drug_id,
                "gene": gene_id,
                "consensus_z_at_ref_dose": consensus_z,
                "se_of_consensus_z": se_z,
                "tau2_heterogeneity": 0.0,
                "n_observations": len(data),
                "converged": True,
                "method": "simple_aggregation",
            }
        except Exception:
            return None

    def _fit_single_point(self, data: pd.DataFrame, drug_id: str, gene_id: str) -> Optional[Dict]:
        try:
            observation = data.iloc[0]
            ref_log_dose = np.log10(self.config.ref_dose)
            dose_penalty = min(1.0, 1.0 / (1.0 + abs(ref_log_dose - observation["log10_dose"])))
            return {
                "drug": drug_id,
                "gene": gene_id,
                "consensus_z_at_ref_dose": float(observation["mean_z"] * dose_penalty),
                "se_of_consensus_z": float(np.sqrt(observation["variance_of_mean_z"])),
                "tau2_heterogeneity": 0.0,
                "n_observations": 1,
                "converged": True,
                "method": "single_point",
            }
        except Exception:
            return None
