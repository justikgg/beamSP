from config import settings as cfg

from core.formula.power_law import PowerLawForm
from core.models.bioinspired.formula_search import FormulaSearchModel
from core.models.bioinspired.optimizers import OPTIMIZERS

import core.models.registry as registry

fixed = {
    "a_h0": 0.0,
    "H": 1.0,
    "s": 1.0,
    "R": 1.0,
    "E": 0.0,
}


class FixedBioDERUK78(FormulaSearchModel):
    name = "bio_de_ruk78_fixed"

    def __init__(self):
        super().__init__(
            optimizer=OPTIMIZERS["de"],
            form=PowerLawForm(fixed_exponents=fixed),
            seed=cfg.SEED,
        )


registry.register_model(FixedBioDERUK78)

from entrypoint._common import run_method

if __name__ == "__main__":
    run_method(
        title="bio_de РУК78: a/h0=0, H=s=R=1, E=0",
        model_names=["bio_de_ruk78_fixed"],
        argv=["--target", "RUK78", "--export"],
    )
