import os
import glob
import json
import numpy as np
from typing import List, Tuple, Dict

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import joblib

from features import PixelFeatureExtractor, MorphFeatureExtractor


DATASET_DIR = os.path.join(os.path.dirname(__file__), "database_parcial")
SAVE_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(SAVE_DIR, exist_ok=True)

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif")


def is_i_label_from_path(path: str) -> int:
    # Considera positivo se a imagem está em 'I_u' (I maiúsculo) ou 'i_l' (i minúsculo)
    parts = os.path.normpath(path).split(os.sep)
    # Busca o diretório da vogal (p.ex. A_u, i_l, etc.)
    # Estrutura esperada: dataset_v20220930/<vogal_dir>/<train_xx>/<arquivo>
    if len(parts) >= 3:
        vowel_dir = parts[-3]  # nome da classe (A_u, e_l, I_u, etc.)
        return 1 if vowel_dir in ("I_u", "i_l") else 0
    # fallback: substring
    return 1 if ("{}I_u{}".format(os.sep, os.sep) in path or "{}i_l{}".format(os.sep, os.sep) in path) else 0


def load_dataset(root: str) -> Tuple[List[str], np.ndarray]:
    # Procura imagens em dataset_v20220930/*/*/*.*
    pattern = os.path.join(root, "*", "*", "*.*")
    files = [p for p in glob.glob(pattern) if os.path.splitext(p)[1].lower() in IMAGE_EXTS]
    files.sort()
    if not files:
        raise RuntimeError(f"Nenhuma imagem encontrada em: {pattern}")

    y = np.array([is_i_label_from_path(p) for p in files], dtype=np.int32)
    return files, y


def main():
    X_paths, y = load_dataset(DATASET_DIR)
    print(f"Imagens carregadas: {len(X_paths)} | Positivos (i): {int(y.sum())} | Negativos (não-i): {int((1-y).sum())}")

    methods: Dict[str, object] = {
        "Pixels + LogisticRegression": make_pipeline(
            PixelFeatureExtractor(size=64),
            StandardScaler(with_mean=True),
            LogisticRegression(max_iter=1000, solver="lbfgs")
        ),
        "Pixels + SVM(RBF)": make_pipeline(
            PixelFeatureExtractor(size=64),
            StandardScaler(with_mean=True),
            SVC(kernel="rbf", C=2.0, gamma="scale", probability=True, class_weight="balanced")
        ),
        "Morfologia + RandomForest": make_pipeline(
            MorphFeatureExtractor(size=64),
            StandardScaler(with_mean=True),
            RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
        ),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
    }

    results_summary = {}
    best_method_name = None
    best_f1 = -1.0

    for name, pipe in methods.items():
        print(f"\n=== Avaliando método: {name} ===")
        cvres = cross_validate(pipe, X_paths, y, cv=cv, scoring=scoring, n_jobs=-1, return_train_score=False)
        avg = {k.replace("test_", ""): float(np.mean(v)) for k, v in cvres.items() if k.startswith("test_")}
        std = {k.replace("test_", ""): float(np.std(v)) for k, v in cvres.items() if k.startswith("test_")}
        results_summary[name] = {"mean": avg, "std": std}

        print(f"Acurácia: {avg['accuracy']:.4f} ± {std['accuracy']:.4f}")
        print(f"Precision: {avg['precision']:.4f} ± {std['precision']:.4f}")
        print(f"Recall:    {avg['recall']:.4f} ± {std['recall']:.4f}")
        print(f"F1:        {avg['f1']:.4f} ± {std['f1']:.4f}")

        if avg["f1"] > best_f1:
            best_f1 = avg["f1"]
            best_method_name = name

    if best_method_name is None:
        raise RuntimeError("Nenhum método avaliado corretamente.")

    print(f"\n>> Melhor método pelo F1: {best_method_name} (F1 médio: {best_f1:.4f})")
    best_pipe = methods[best_method_name]

    # Treina com todo o dataset e salva
    print("Treinando o melhor método em todo o dataset...")
    best_pipe.fit(X_paths, y)
    model_path = os.path.join(SAVE_DIR, "is_i_best.joblib")
    joblib.dump({"pipeline": best_pipe, "method_name": best_method_name}, model_path)
    print(f"Modelo salvo em: {model_path}")

    # Exporta um relatório JSON com as métricas (para anexar ao relatório do trabalho)
    report_path = os.path.join(SAVE_DIR, "cv_results.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({"results": results_summary, "best_method": best_method_name}, f, ensure_ascii=False, indent=2)
    print(f"Relatório de validação cruzada salvo em: {report_path}")


if __name__ == "__main__":
    main()