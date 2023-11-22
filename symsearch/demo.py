import os
import pandas as pd

from flask import Flask, request, jsonify
from .features.faiss import CustomFAISS
from .inference import RetrieveInference

app = Flask(__name__)


CKPT_PATH = "ckpt"
embedding_fn = RetrieveInference(CKPT_PATH, p_max_length=256)
faiss = None


def init_model():
    global faiss
    df_doanhnghiep = load_csv_file(
        os.path.join("data", "danh-muc-nganh-nghe-dang-ky-kinh-doanh.xlsx")
    )
    print("\nWaiting load database . . .\n")
    faiss = add_to_db(df_doanhnghiep)


def load_csv_file(path: str = "danh-muc-nganh-nghe-dang-ky-kinh-doanh.xlsx"):
    df = pd.read_excel(path)
    df["id"] = df[df.columns[0:5]].apply(
        lambda row: "-".join(row.fillna("").values.astype(str)), axis=1
    )
    df["id"] = df["id"].str.replace(r"\-+", "-", regex=True)
    df["id"] = (
        df["id"]
        .str.replace(r"^[-]*", "", regex=True)
        .str.replace(r"-$", "", regex=True)
    )
    df["id"] = df["id"].str.replace(r"(\d+)\.0", r"\1", regex=True)
    df = df.drop(columns=list(df.columns[0:5]))

    return df


def add_to_db(df: pd.DataFrame):
    faiss = CustomFAISS.from_texts(
        texts=df["Tên ngành"].tolist(), ids=df["id"].tolist(), embedding=embedding_fn
    )
    return faiss


@app.route("/retrieval", methods=["POST"])
def query_result():
    req = request.json
    result = faiss.similarity_search_with_score(req["query"], int(req["knn"]))
    print(result)
    return jsonify(result)


if __name__ == "__main__":
    init_model()
    app.run(host="0.0.0.0", port=5000)
