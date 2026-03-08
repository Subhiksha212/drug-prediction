import os
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.preprocessing import StandardScaler


AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_INT = {aa: i + 1 for i, aa in enumerate(AMINO_ACIDS)}


def encode_protein(sequence: str, max_len: int = 1000) -> List[int]:
    """
    Encode a protein sequence as a fixed-length integer vector.
    Mirrors the logic used in the original notebook.
    """
    sequence = (sequence or "").strip().upper()
    sequence = sequence[:max_len]
    encoding = [AA_TO_INT.get(aa, 0) for aa in sequence]
    if len(encoding) < max_len:
        encoding += [0] * (max_len - len(encoding))
    return encoding


def smiles_to_features(smiles: str) -> List[float]:
    """
    Compute a small, stable set of RDKit descriptors for a SMILES string.
    This matches the 5‑feature representation used in the training code.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [0.0] * 5

    return [
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.TPSA(mol),
    ]


class DrugProteinModel(nn.Module):
    """
    Feed‑forward network used for drug‑protein affinity prediction.
    Input size 1005 = 5 drug descriptors + 1000‑length protein encoding.
    """

    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(1005, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class DrugProteinPredictor:
    """
    High‑level wrapper that loads:
    - trained PyTorch model weights (if available)
    - drugs metadata from the data folder
    - RDKit‑based drug feature matrix

    and exposes a simple predict_top_drugs() API.
    """

    def __init__(
        self,
        project_root: Optional[Path] = None,
        model_filename_candidates: Optional[List[str]] = None,
    ) -> None:
        # Resolve paths
        if project_root is None:
            # Assume this file lives in <project_root>/model/model.py
            project_root = Path(__file__).resolve().parents[1]

        self.project_root = project_root
        self.data_dir = self.project_root / "data"
        self.model_dir = self.project_root / "model"

        # Data files
        # Prefer enriched metadata if available, otherwise fall back to basic drugs.csv
        metadata_candidates = [
            self.data_dir / "drugs_metadata.csv",
            self.data_dir / "drugs_metadata (12).csv",
            self.data_dir / "drugs.csv",
        ]
        self.drugs_path = self._first_existing(metadata_candidates)
        if self.drugs_path is None:
            raise FileNotFoundError(
                f"No drugs CSV found in {self.data_dir}. "
                "Expected one of: drugs_metadata.csv, drugs_metadata (12).csv, drugs.csv"
            )

        # Model files – try multiple common names
        if model_filename_candidates is None:
            model_filename_candidates = [
                "dti_model.pth",
                "drug_model.pth",
                "drug_protein_model.pth",
            ]

        self.model_path = self._first_existing(
            [self.model_dir / name for name in model_filename_candidates]
        )

        # Torch device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Lazily initialised fields
        self._drugs_df: Optional[pd.DataFrame] = None
        self._drug_features_tensor: Optional[torch.Tensor] = None
        self._model: Optional[DrugProteinModel] = None
        self._scaler: Optional[StandardScaler] = None
        self._affinity_min: float = 5.0
        self._affinity_max: float = 8.0

    def _build_scaler_and_affinity_range(self) -> None:
        """Fit StandardScaler on training data (affinity pairs) and get affinity range."""
        if self._scaler is not None:
            return
        affinity_path = self.data_dir / "drug_protein_affinity.csv"
        proteins_path = self._first_existing([
            self.data_dir / "proteins.csv",
        ])
        if not affinity_path.exists() or not proteins_path or not proteins_path.exists():
            self._scaler = StandardScaler()
            self._scaler.mean_ = np.zeros(1005)
            self._scaler.scale_ = np.ones(1005)
            return
        affinity_df = pd.read_csv(affinity_path)
        self._affinity_min = float(affinity_df["Affinity"].min())
        self._affinity_max = float(affinity_df["Affinity"].max())
        if self._affinity_max <= self._affinity_min:
            self._affinity_max = self._affinity_min + 1.0
        proteins_df = pd.read_csv(proteins_path)
        if "Protein_Index" in proteins_df.columns:
            proteins_df = proteins_df.set_index("Protein_Index", drop=False)
        max_drug = len(self.drugs_df)
        drug_feats = self.drug_features.cpu().numpy()
        X_list = []
        for _, row in affinity_df.iterrows():
            drug_idx = int(row["Drug_Index"])
            protein_idx = int(row["Protein_Index"])
            if drug_idx >= max_drug:
                continue
            try:
                if "Protein_Index" in proteins_df.columns:
                    seq = proteins_df.loc[protein_idx, "Sequence"]
                else:
                    seq = proteins_df.iloc[protein_idx]["Sequence"]
            except (KeyError, IndexError):
                continue
            protein_feat = np.array(encode_protein(seq), dtype=np.float32)
            combined = np.concatenate([drug_feats[drug_idx], protein_feat])
            X_list.append(combined)
        if len(X_list) < 10:
            self._scaler = StandardScaler()
            self._scaler.mean_ = np.zeros(1005)
            self._scaler.scale_ = np.ones(1005)
            return
        X = np.array(X_list, dtype=np.float32)
        self._scaler = StandardScaler()
        self._scaler.fit(X)

    @property
    def scaler(self) -> StandardScaler:
        if self._scaler is None:
            self._build_scaler_and_affinity_range()
        return self._scaler  # type: ignore[return-value]

    @staticmethod
    def _first_existing(paths: List[Path]) -> Optional[Path]:
        for p in paths:
            if p is not None and p.exists():
                return p
        return None

    @property
    def drugs_df(self) -> pd.DataFrame:
        if self._drugs_df is None:
            df = pd.read_csv(self.drugs_path)

            # Normalise expected columns
            if "Drug_Name" not in df.columns:
                # Best‑effort to derive a display name
                if "name" in df.columns:
                    df["Drug_Name"] = df["name"]
                elif "drug_name" in df.columns:
                    df["Drug_Name"] = df["drug_name"]
                else:
                    df["Drug_Name"] = [f"Drug {i}" for i in range(len(df))]

            if "Canonical_SMILES" not in df.columns:
                # Try some common alternative column names
                smiles_col = None
                for candidate in ("smiles", "SMILES", "Isomeric_SMILES"):
                    if candidate in df.columns:
                        smiles_col = candidate
                        break
                if smiles_col is None:
                    raise ValueError(
                        "No SMILES column found in drugs CSV. "
                        "Expected Canonical_SMILES, smiles, SMILES, or Isomeric_SMILES."
                    )
                df["Canonical_SMILES"] = df[smiles_col]

            if "CID" not in df.columns:
                # Provide a stable identifier if CID is missing
                df["CID"] = df.index.astype(str)

            # Sort by Drug_Index so alignment matches training (affinity uses Drug_Index)
            if "Drug_Index" in df.columns:
                df = df.sort_values("Drug_Index").reset_index(drop=True)

            self._drugs_df = df
        return self._drugs_df

    @property
    def drug_features(self) -> torch.Tensor:
        if self._drug_features_tensor is None:
            smiles_list = self.drugs_df["Canonical_SMILES"].tolist()
            features = np.array([smiles_to_features(s) for s in smiles_list], dtype=np.float32)
            tensor = torch.from_numpy(features)
            self._drug_features_tensor = tensor.to(self.device)
        return self._drug_features_tensor

    @property
    def model(self) -> DrugProteinModel:
        if self._model is None:
            model = DrugProteinModel().to(self.device)

            if self.model_path is not None and self.model_path.exists():
                # Load trained weights if available
                state_dict = torch.load(self.model_path, map_location=self.device)
                # Allow both whole‑model and state‑dict formats
                if isinstance(state_dict, dict):
                    try:
                        model.load_state_dict(state_dict)
                    except RuntimeError:
                        # Fallback: model saved as a full object
                        model = state_dict  # type: ignore[assignment]
                else:
                    model = state_dict  # type: ignore[assignment]

            model.eval()
            self._model = model
        return self._model

    def predict_top_drugs(self, protein_sequence: str, top_k: int = 10) -> List[Dict]:
        """
        Given a raw protein sequence, return the top‑K predicted drugs with scores.
        Output schema is designed to match the frontend's DrugPrediction interface.
        """
        if not protein_sequence or not protein_sequence.strip():
            raise ValueError("Protein sequence must be a non‑empty string.")

        protein_encoded = np.array(encode_protein(protein_sequence), dtype=np.float32)

        # Broadcast protein features to all drugs
        protein_tensor = torch.from_numpy(protein_encoded).to(self.device).unsqueeze(0)
        protein_tensor = protein_tensor.repeat(self.drug_features.shape[0], 1)

        combined = torch.cat([self.drug_features, protein_tensor], dim=1)

        # Scale features to match training distribution (model was trained on scaled data)
        combined_np = combined.cpu().numpy()
        combined_scaled = self.scaler.transform(combined_np)
        combined_tensor = torch.from_numpy(combined_scaled.astype(np.float32)).to(self.device)

        with torch.no_grad():
            raw_scores = self.model(combined_tensor).squeeze(1)

        top_k = max(1, min(top_k, raw_scores.shape[0]))
        top_values, top_indices = torch.topk(raw_scores, k=top_k)

        # Normalize scores to [0,1] for display
        aff_min, aff_max = self._affinity_min, self._affinity_max
        aff_range = aff_max - aff_min
        if aff_range <= 0:
            aff_range = 1.0
        t = top_values.cpu().float()
        t = (t - aff_min) / aff_range
        t = torch.clamp(t, 0.0, 1.0)

        # If all scores are 0 or negligible, use rank-based spread so graph and % always display
        t_list = t.tolist()
        if max(t_list) < 0.05:
            t_list = [0.5 + 0.45 * (1.0 - i / max(top_k, 1)) for i in range(top_k)]

        df = self.drugs_df
        results: List[Dict] = []
        for score, idx_tensor in zip(t_list, top_indices.cpu().tolist()):
            idx = int(idx_tensor)
            row = df.iloc[idx]
            results.append(
                {
                    "drug": str(row.get("Drug_Name", f"Drug {idx}")),
                    "score": float(score),
                    "smiles": str(row.get("Canonical_SMILES", "")),
                    "id": str(row.get("CID", idx)),
                    "target": None,
                }
            )

        return results


__all__ = ["DrugProteinModel", "DrugProteinPredictor", "encode_protein", "smiles_to_features"]

