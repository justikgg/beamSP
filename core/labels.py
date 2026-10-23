GROUP_LABELS = {
    "linear": "Линейные",
    "kernel": "Ядровые методы (SVR / GPR)",
    "neighbors": "Метод ближайших соседей",
    "ensemble": "Ансамбли деревьев",
    "formula": "Вывод формулы (символьная / биоинспир.)",
    "neural": "Нейросети",
    "experimental": "Экспериментальные",
}

GROUP_ORDER = ["linear", "kernel", "neighbors", "ensemble", "formula", "neural", "experimental"]

TARGET_LABELS = {
    "SP63": "СП 63.13330",
    "RUK78": "Руководство 1978",
}

MODEL_LABELS = {
    "linear": "Линейная регрессия",
    "ridge": "Ridge-регрессия",
    "lasso": "Lasso-регрессия",
    "elasticnet": "ElasticNet-регрессия",
    "bio_ga": "Степенная: генетический алгоритм",
    "bio_de": "Степенная: дифференциальная эволюция",
    "bio_pso": "Степенная: рой частиц (PSO)",
    "bio_cmaes": "Степенная: CMA-ES",
    "log_linear_ls": "Степенная: ограниченный Log-Linear Least Squares",
    "log_linear_trf": "Степенная: Log-Linear LS + TRF",
    "gbr": "Градиентный бустинг",
    "svr": "Метод опорных векторов (SVR)",
    "knn": "Метод ближайших соседей (KNN)",
    "gpr": "Гауссовский процесс (GPR)",
    "symreg": "Символьная регрессия",
    "bayes_symreg": "Байесовская символьная регрессия (ARD)",
    "pinn": "Физически-информированная нейросеть (PINN)",
    "mlp": "Многослойный перцептрон (MLP)",
    "rl_ac": "RL (актор-критик)",
}

Q_UNIT = "кН"

def method_label(name):
    if name.endswith("_fixed") and name[:-6] in MODEL_LABELS:
        return MODEL_LABELS[name[:-6]] + " (фикс.)"
    return MODEL_LABELS.get(name, name)


def target_label(name):
    return TARGET_LABELS.get(name, name)
