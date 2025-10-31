import os
import sys
import joblib
import numpy as np

def main():
    if len(sys.argv) < 2:
        print("Uso: python infer_is_i.py <caminho_da_imagem>")
        sys.exit(1)

    img_path = sys.argv[1]
    if not os.path.isfile(img_path):
        print(f"Arquivo não encontrado: {img_path}")
        sys.exit(1)

    model_path = os.path.join(os.path.dirname(__file__), "models", "is_i_best.joblib")
    if not os.path.isfile(model_path):
        print(f"Modelo não encontrado em {model_path}. Rode primeiro: train_eval.py")
        sys.exit(1)

    payload = joblib.load(model_path)
    pipe = payload["pipeline"]
    method_name = payload.get("method_name", "Desconhecido")

    # Predição
    X = [img_path]
    y_pred = pipe.predict(X)
    proba = None
    if hasattr(pipe, "predict_proba"):
        # Alguns classificadores não expõem probability direto na pipeline, então tentamos no último passo
        try:
            proba = pipe.predict_proba(X)[:, 1]
        except Exception:
            proba = None

    print(f"Método carregado: {method_name}")
    if y_pred[0] == 1:
        if proba is not None:
            print(f"Resposta: É 'i' (confiança: {proba[0]:.3f})")
        else:
            print("Resposta: É 'i'")
    else:
        if proba is not None:
            print(f"Resposta: NÃO é 'i' (confiança: {1.0 - proba[0]:.3f})")
        else:
            print("Resposta: NÃO é 'i'")

if __name__ == "__main__":
    main()