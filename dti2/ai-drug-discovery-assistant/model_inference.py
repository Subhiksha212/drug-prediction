import os
from typing import List, Dict, Any

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import Descriptors


amino_acids = "ACDEFGHIKLMNPQRSTVWY"
aa_to_int = {aa: i + 1 for i, aa in enumerate(amino_acids)}


def encode_protein(seq: str, max_len: int = 1000) -> np.ndarray:
    """Encode an amino-acid sequence to a fixed-length numeric vector."""
    seq = (seq or "").upper().strip()[:max_len]
    encoding = [aa_to_int.get(aa, 0) for aa in seq]
    if len(encoding) < max_len:
        encoding += [0] * (max_len - len(encoding))
    return np.asarray(encoding, dtype=np.float32)


def smiles_to_features(smiles: str) -> List[float] | None:
    """Compute simple RDKit descriptors for a SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    return [
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.TPSA(mol),
    ]


class DrugProteinModel(torch.nn.Module):
    """Model architecture copied from the original training script."""

    def __init__(self) -> None:
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(1005, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.network(x)


class DTIModelService:
    """Thin wrapper around the PyTorch model for inference."""

    def __init__(self, model_path: str, device: str | None = None) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = DrugProteinModel().to(self.device)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")

        state_obj = torch.load(model_path, map_location=self.device)

        # The training script saved a state_dict; fall back to full-model load if needed.
        try:
            self.model.load_state_dict(state_obj)
        except Exception:
            # If a full model was saved instead of a state_dict
            if isinstance(state_obj, torch.nn.Module):
                self.model = state_obj.to(self.device)
            else:
                raise

        self.model.eval()

    def predict_for_sequence(
        self,
        sequence: str,
        drugs: List[Dict[str, Any]],
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """Score a list of drugs for a given protein sequence."""
        if not sequence or not sequence.strip():
            return []

        protein_vec = encode_protein(sequence)  # (1000,)
        protein_tensor = torch.from_numpy(protein_vec).unsqueeze(0)  # (1, 1000)

        drug_features: list[list[float]] = []
        valid_drugs: list[Dict[str, Any]] = []

        for drug in drugs:
            smiles = drug.get("smiles")
            if not smiles:
                continue
            feats = smiles_to_features(smiles)
            if feats is None:
                continue
            drug_features.append(feats)
            valid_drugs.append(drug)

        if not drug_features:
            return []

        drug_tensor = torch.tensor(drug_features, dtype=torch.float32)  # (N, 5)
        protein_tensor = protein_tensor.repeat(drug_tensor.shape[0], 1)  # (N, 1000)

        combined = torch.cat((drug_tensor, protein_tensor), dim=1).to(self.device)  # (N, 1005)

        with torch.no_grad():
            # Keep scores in [0, 1] using sigmoid as in the notebook
            scores = torch.sigmoid(self.model(combined)).squeeze().cpu().numpy()

        results: list[Dict[str, Any]] = []
        for drug, score in zip(valid_drugs, scores):
            result = dict(drug)
            result["score"] = float(score)
            results.append(result)

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

