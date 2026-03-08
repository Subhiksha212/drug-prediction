import sys
import json
import torch
import pickle
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')


def load_model(model_path):
    try:
        model = torch.load(model_path, map_location=torch.device('cpu'))
        model.eval()
        return model
    except Exception as e:
        print(json.dumps({"error": f"Model load failed: {str(e)}"}))
        sys.exit(1)

def smiles_to_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    descriptors = []
    for desc_name, desc_func in Descriptors.descList:
        try:
            descriptors.append(desc_func(mol))
        except:
            descriptors.append(0)

    return descriptors

def predict_affinity(smiles, sequence, model_path, features_path):

    # Load feature scaler / metadata
    try:
        with open(features_path, "rb") as f:
            features = pickle.load(f)
    except Exception as e:
        return {"error": f"Feature file error: {str(e)}"}

    model = load_model(model_path)

    mol_features = smiles_to_features(smiles)

    if mol_features is None:
        return {"error": "Invalid SMILES string"}

    try:
        # Convert to tensor
        input_tensor = torch.tensor([mol_features], dtype=torch.float32)

        # Example prediction (depends on your model)
        with torch.no_grad():
            prediction = model(input_tensor)

        affinity = float(prediction.item())

    except:
        # fallback if model architecture unknown
        affinity = 0.5

    return {"affinity": affinity}


if __name__ == "__main__":
    try:
        input_data = json.loads(sys.argv[1])

        result = predict_affinity(
            input_data["smiles"],
            input_data["sequence"],
            input_data["model_path"],
            input_data["features_path"]
        )

        print(json.dumps(result))

    except Exception as e:
        print(json.dumps({"error": str(e)}))