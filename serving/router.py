# serving/router.py
import random
from typing import Tuple
from dataclasses import dataclass

@dataclass
class ExperimentConfig:
    """
    Defines an A/B experiment.
    traffic_split = % of traffic going to model_a
    e.g. 0.7 means 70% → model_a, 30% → model_b
    """
    experiment_id: str
    model_a_id:    str
    model_b_id:    str
    traffic_split: float   # 0.0 to 1.0 for model_a
    is_active:     bool = True
    description:   str  = ""

class ABRouter:
    """
    Routes incoming requests to model_a or model_b
    based on the configured traffic split.

    This is the core of A/B testing:
    - Each request gets randomly assigned
    - Assignment is logged with the prediction
    - We can later compare model_a vs model_b results
    """

    def __init__(self):
        self.experiments: dict[str, ExperimentConfig] = {}

    def create_experiment(self, config: ExperimentConfig):
        self.experiments[config.experiment_id] = config
        print(f"🧪 Experiment created: {config.experiment_id} "
              f"| Split: {config.traffic_split:.0%} → {config.model_a_id}, "
              f"{1 - config.traffic_split:.0%} → {config.model_b_id}")

    def route(self, experiment_id: str) -> Tuple[str, str]:
        """
        Returns (model_id, variant) for this request.
        variant is 'A' or 'B' — used for tracking.

        This random assignment is what makes it A/B testing:
        each request independently gets assigned, creating
        two unbiased groups over time.
        """
        exp = self.experiments.get(experiment_id)
        if not exp or not exp.is_active:
            # No experiment — just use model_a
            return "model_a", "A"

        if random.random() < exp.traffic_split:
            return exp.model_a_id, "A"
        else:
            return exp.model_b_id, "B"

    def update_split(self, experiment_id: str,
                     new_split: float):
        """
        Change traffic split on the fly.
        e.g. Start 50/50, then move to 80/20 as confidence grows.
        """
        if experiment_id in self.experiments:
            self.experiments[experiment_id].traffic_split = (
                new_split
            )

    def stop_experiment(self, experiment_id: str,
                        winner: str):
        """
        End experiment and route 100% to winner.
        """
        if experiment_id in self.experiments:
            exp = self.experiments[experiment_id]
            exp.is_active = False
            if winner == "A":
                exp.traffic_split = 1.0
            else:
                exp.traffic_split = 0.0
            print(f"🏆 Experiment {experiment_id} ended. "
                  f"Winner: Model {winner}")

# Singleton router
router = ABRouter()
router.create_experiment(ExperimentConfig(
    experiment_id= "exp_001",
    model_a_id=    "model_a",
    model_b_id=    "model_b",
    traffic_split= 0.5,       # 50/50 split to start
    description=   "XGBoost v1 vs LightGBM v2"
))