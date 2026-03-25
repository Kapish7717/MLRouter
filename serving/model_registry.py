# serving/model_registry.py
import pickle, json, os
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class RegisteredModel:
    model_id:   str          # "model_a" or "model_b"
    model:      Any          # the actual sklearn/xgb object
    metadata:   Dict         # accuracy, auc, version etc
    version:    str          # version tag (extracted from metadata)
    is_active:  bool = True

class ModelRegistry:
    """
    Central store for all loaded models.
    In production this would be MLflow or a database.
    Here we keep it simple with in-memory loading.
    """

    def __init__(self):
        self.models: Dict[str, RegisteredModel] = {}

    def register(self, model_id: str, model_path: str,
                 metadata_path: str):
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        with open(metadata_path) as f:
            metadata = json.load(f)

        self.models[model_id] = RegisteredModel(
            model_id=model_id,
            model=model,
            metadata=metadata,
            version=metadata.get("version", "unknown")
        )
        print(f"✅ Registered: {model_id} "
              f"(v{metadata.get('version', '?')})")

    def get(self, model_id: str) -> RegisteredModel:
        if model_id not in self.models:
            raise ValueError(f"Model '{model_id}' not found")
        return self.models[model_id]

    def list_models(self):
        return [
            {
                "model_id":   m.model_id,
                "model_type": m.metadata.get("model_type"),
                "version":    m.metadata.get("version"),
                "accuracy":   m.metadata.get("accuracy"),
                "roc_auc":    m.metadata.get("roc_auc"),
                "is_active":  m.is_active,
            }
            for m in self.models.values()
        ]

# Singleton — one registry for the whole app
registry = ModelRegistry()
registry.register(
    "model_a",
    "models/model_a.pkl",
    "models/model_a_metadata.json"
)
registry.register(
    "model_b",
    "models/model_b.pkl",
    "models/model_b_metadata.json"
)

def _main_():
    registered_model = registry.get("model_a")
    print(registered_model)

if __name__ == "__main__":
    _main_()