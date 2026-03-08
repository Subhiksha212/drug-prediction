from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import os
import sys
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd

# Get the directory where this script is located
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent.parent  # c:/.../drug-ai-app
DATA_DIR = PROJECT_ROOT / "data"

# Ensure project root is on Python path so we can import the model package
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from model.model import DrugProteinPredictor  # type: ignore  # noqa: E402

app = Flask(__name__, static_folder="dist", static_url_path="")
CORS(app)  # Enable CORS for all routes

# Instantiate the predictor once at startup and reuse it for all requests
PREDICTOR = DrugProteinPredictor(project_root=PROJECT_ROOT)
_ = PREDICTOR.drug_features  # Eagerly load RDKit features so features_loaded is true at startup

# Build a lightweight drug dataset view for list/search/detail endpoints
DRUG_DATASET: List[Dict] = []
for _, row in PREDICTOR.drugs_df.iterrows():
    DRUG_DATASET.append(
        {
            "drug": str(row.get("Drug_Name", "Unknown")),
            "smiles": str(row.get("Canonical_SMILES", "")),
            "id": str(row.get("CID", "")),
        }
    )

# Dataset stats for the status endpoint
PROTEINS_CSV = DATA_DIR / "proteins.csv"
AFFINITY_CSV = DATA_DIR / "drug_protein_affinity.csv"

try:
    TOTAL_PROTEINS = len(pd.read_csv(PROTEINS_CSV)) if PROTEINS_CSV.exists() else 0
except Exception:
    TOTAL_PROTEINS = 0

try:
    TOTAL_INTERACTIONS = len(pd.read_csv(AFFINITY_CSV)) if AFFINITY_CSV.exists() else 0
except Exception:
    TOTAL_INTERACTIONS = 0

# --- API routes (must be registered before the catch-all below) ---

@app.route('/api/status', methods=['GET'])
def get_status():
    """Get server status and stats"""
    return jsonify({
        "status": "online",
        "totalDrugs": len(DRUG_DATASET),
        "totalProteins": TOTAL_PROTEINS,
        "totalInteractions": TOTAL_INTERACTIONS,
        "isLoaded": True,
        "timestamp": pd.Timestamp.utcnow().isoformat()
    })

@app.route('/api/drugs', methods=['GET'])
def get_drugs():
    """Get all drugs or search by query"""
    search = request.args.get('search', '')
    limit = request.args.get('limit')
    
    if search:
        # Search drugs by name
        search_lower = search.lower()
        results = [d for d in DRUG_DATASET if search_lower in d['drug'].lower()]
    else:
        results = DRUG_DATASET
    
    if limit:
        try:
            results = results[:int(limit)]
        except:
            pass
    
    return jsonify(results)

@app.route('/api/drugs/<drug_id>', methods=['GET'])
def get_drug_details(drug_id):
    """Get details for a specific drug"""
    drug = next((d for d in DRUG_DATASET if d['id'] == drug_id), None)
    if not drug:
        return jsonify({"error": "Drug not found"}), 404
    
    # Add mock interactions
    drug_copy = drug.copy()
    drug_copy['molecularWeight'] = f"{np.random.randint(300, 600)} g/mol"
    drug_copy['clinicalPhase'] = np.random.choice(['Phase I', 'Phase II', 'Phase III'])
    drug_copy['toxicityScore'] = f"{np.random.uniform(0.1, 0.5):.3f}"
    
    return jsonify(drug_copy)

@app.route('/api/predict', methods=['POST'])
def predict_drugs():
    """Predict drug-protein interactions"""
    data = request.get_json()
    sequence = (data or {}).get('sequence')
    
    if not sequence:
        return jsonify({"error": "Sequence is required"}), 400
    
    if len(sequence) < 5:
        return jsonify({"error": "Sequence must be at least 5 characters"}), 400
    
    try:
        predictions = PREDICTOR.predict_top_drugs(sequence, top_k=10)
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    return jsonify(predictions)

@app.route('/api/test', methods=['GET'])
def test():
    """Test endpoint to verify server is running"""
    return jsonify({
        "message": "Flask server is running",
        "drugs_loaded": len(DRUG_DATASET),
        "model_exists": PREDICTOR.model_path is not None,
        "features_loaded": bool(PREDICTOR._drug_features_tensor is not None)
    })

# --- Static / SPA catch-all (register last so /api/* is matched above) ---
DIST_DIR = BASE_DIR / "dist"

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    """Serve React app from dist/ or a simple info page if dist is missing."""
    index_path = DIST_DIR / "index.html"
    if path and (DIST_DIR / path).is_file():
        return send_from_directory(str(DIST_DIR), path)
    if index_path.is_file():
        return send_from_directory(str(DIST_DIR), "index.html")
    # No build yet: show backend info so opening http://localhost:5000/ doesn't 404
    return (
        "<!DOCTYPE html><html><head><meta charset='UTF-8'><title>Drug Discovery API</title></head><body>"
        "<h1>Drug Discovery Backend</h1><p>API is running. Use the frontend (e.g. <code>npm run dev</code>) "
        "or call:</p><ul>"
        "<li><a href='/api/test'>GET /api/test</a> – test</li>"
        "<li><a href='/api/status'>GET /api/status</a> – status</li>"
        "<li>POST /api/predict – predict (body: <code>{\"sequence\": \"...\"}</code>)</li>"
        "</ul></body></html>",
        200,
        {"Content-Type": "text/html; charset=utf-8"},
    )

if __name__ == '__main__':
    print(f"\n🚀 Starting Flask server...")
    print(f"📁 Base directory: {BASE_DIR}")
    print(f"📁 Project root: {PROJECT_ROOT}")
    print(f"📁 Data directory: {DATA_DIR}")
    print(f"💊 Drugs loaded: {len(DRUG_DATASET)}")
    print(f"📊 Model path: {PREDICTOR.model_path}")
    print(f"\n🌐 Server running on http://localhost:5000")
    print(f"📡 API endpoints:")
    print(f"   GET  /api/test - Test connection")
    print(f"   GET  /api/status - Server status")
    print(f"   GET  /api/drugs - List drugs")
    print(f"   GET  /api/drugs/<id> - Drug details")
    print(f"   POST /api/predict - Predict interactions\n")
    
    app.run(debug=True, port=5000, host='0.0.0.0')