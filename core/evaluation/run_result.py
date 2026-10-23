from dataclasses import dataclass, field


@dataclass
class RunResult:
    metrics: dict = field(default_factory=dict)
    predictions: dict = field(default_factory=dict)
    formulas: dict = field(default_factory=dict)
    fold_rmse: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
