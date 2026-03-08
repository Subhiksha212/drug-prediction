import os
from datetime import datetime, timezone

from flask import Flask, jsonify, request
from flask_cors import CORS

from model_inference import DTIModelService


app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})


# Sample drug dataset (replace with your actual dataset)
DRUG_DATASET = [
    {"id": "1", "drug": "Remdesivir", "smiles": "CCC(CC)COC(=O)[C@H](C)N[P@](=O)(OC[C@H]1O[C@@](C#N)([C@H](O)[C@@H]1O)C1=CC=C2N1N=NN2)OC1=CC=CC=C1"},
    {"id": "2", "drug": "Favipiravir", "smiles": "C1=C(N=C(C(=O)N1)C(=O)N)C(F)(F)F"},
    {"id": "3", "drug": "Hydroxychloroquine", "smiles": "CCN(CCO)CCCC(C)NC1=C2C=CC(=CC2=NC=C1)Cl"},
    {"id": "4", "drug": "Oseltamivir", "smiles": "CCOC(=O)C1=C[C@@H](OC(CC)CC)[C@H](NC(C)=O)[C@@H](N)C1"},
    {"id": "5", "drug": "Lopinavir", "smiles": "CC1=C(C(=CC=C1)C)OCC(=O)NC(CC2=CC=CC=C2)C(CC(CC3=CC=CC=C3)NC(=O)C(C(C)C)N4CCCNC4=O)O"},
    {"id": "6", "drug": "Ritonavir", "smiles": "CC(C)C1=NC(=CS1)CN(C)C(=O)NC(C(C)C)C(=O)NC(CC2=CC=CC=C2)CC(C(CC3=CC=CC=C3)NC(=O)OCC4=CN=CS4)O"},
    {"id": "7", "drug": "Azithromycin", "smiles": "CC[C@H]1OC(=O)[C@H](C)[C@@H](O[C@H]2C[C@@](C)(OC)[C@@H](O)[C@H](C)O2)[C@H](C)[C@@H](O[C@@H]3O[C@H](C)C[C@@H]([C@H]3O)N(C)C)[C@](C)(O)C[C@@H](C)C(=O)[C@H](C)[C@@H](O)[C@]1(C)O"},
    {"id": "8", "drug": "Dexamethasone", "smiles": "C[C@@H]1C[C@H]2[C@@H]3CCC4=CC(=O)C=C[C@]4(C)[C@@]3(F)[C@@H](O)C[C@]2(C)[C@@]1(O)C(=O)CO"},
    {"id": "9", "drug": "Ivermectin", "smiles": "CC(C)CC1C(C)C2C3C(C=C4C(C)C(CCC5C(C)C(OC)CC45)OC)C(OC)C(OC)C3C(OC)C(OC)C2C(OC)C(OC)C1OC"},
    {"id": "10", "drug": "Chloroquine", "smiles": "CCN(CC)CCCC(C)NC1=C2C=CC(=CC2=NC=C1)Cl"},
]


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(PROJECT_ROOT, "model", "drug_model.pth")

try:
    dti_service = DTIModelService(MODEL_PATH)
    MODEL_LOADED = True
except Exception as exc:  # pragma: no cover - defensive logging
    print(f"Failed to load DTI model from {MODEL_PATH}: {exc}")
    dti_service = None
    MODEL_LOADED = False


@app.route("/api/status", methods=["GET"])
def get_status():
    return jsonify(
        {
            "status": "online" if MODEL_LOADED else "degraded",
            "totalDrugs": len(DRUG_DATASET),
            "totalProteins": 1,  # Placeholder
            "totalInteractions": len(DRUG_DATASET),
            "isLoaded": bool(MODEL_LOADED),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    )


@app.route("/api/predict", methods=["POST"])
def predict_drugs():
    if not MODEL_LOADED or dti_service is None:
        return jsonify({"error": "Model is not loaded on the server"}), 500

    data = request.get_json(silent=True) or {}
    sequence = data.get("sequence")

    if not sequence or not isinstance(sequence, str):
        return jsonify({"error": "Sequence is required"}), 400

    try:
        predictions = dti_service.predict_for_sequence(sequence, DRUG_DATASET, top_k=10)
    except Exception as exc:  # pragma: no cover - defensive logging
        return jsonify({"error": f"Prediction failed: {exc}"}), 500

    # Ensure consistent shape for the frontend
    response = [
        {
            "id": item.get("id") or str(idx + 1),
            "drug": item.get("drug", f"Drug {idx + 1}"),
            "smiles": item.get("smiles", ""),
            "score": float(item.get("score", 0.0)),
        }
        for idx, item in enumerate(predictions)
    ]

    return jsonify(response)


@app.route("/api/drugs/<drug_id>", methods=["GET"])
def get_drug_details(drug_id: str):
    """Minimal details endpoint used by the frontend modal."""
    drug = next((d for d in DRUG_DATASET if d.get("id") == drug_id), None)
    if not drug:
        return jsonify({"error": "Drug not found"}), 404

    # Basic placeholder metadata; can be extended with real data later.
    return jsonify(
        {
            "id": drug["id"],
            "drug": drug["drug"],
            "smiles": drug["smiles"],
            "score": 0.0,
            "molecularWeight": None,
            "drugClass": "Candidate compound",
            "mechanism": "Mechanism of action not specified for this demo dataset.",
            "clinicalPhase": "In Silico",
            "toxicityScore": None,
            "interactions": [],
        }
    )


@app.route("/api/test", methods=["GET"])
def test():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
