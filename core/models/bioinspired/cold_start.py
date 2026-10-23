from config import settings as cfg

from core.formula.power_law import PowerLawForm
from core.models.bioinspired.formula_search import FormulaSearchModel
from core.models.bioinspired.optimizers import OPTIMIZERS

import core.models.registry as registry

fixed = {
    "H": 1.0,
    "s": 1.0,
    "R": 1.0,
    "E": 0.0,
}


class FixedBioDENoE(FormulaSearchModel):
    name = "bio_de_no_e"

    def __init__(self):
        super().__init__(
            optimizer=OPTIMIZERS["de"],
            form=PowerLawForm(fixed_exponents=fixed),
            seed=cfg.SEED,
        )


registry.register_model(FixedBioDENoE)

from entrypoint._common import run_method

if __name__ == "__main__":
    run_method(
        title="bio_de без E, зафиксированы H/s/R",
        model_names=["bio_de_no_e"],
    )