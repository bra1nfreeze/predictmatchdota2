from flask import Flask, render_template, request
import pandas as pd
import xgboost as xgb

app = Flask(__name__)

# Load data
heroes_df = pd.read_csv("lanjutkan/80-20/data/heroes.csv")
items_df = pd.read_csv("lanjutkan/80-20/data/items.csv")
model = xgb.Booster()
model.load_model("lanjutkan/80-20/model/xgboost_model.json")


def get_hero_options():
    return [
        {"hero_id": row["hero_id"], "localized_name": row["localized_name"]}
        for _, row in heroes_df.iterrows()
    ]
def get_item_options():
    return [
        {"item_id": row["item_id"], "item_name": row["item_name"]}
        for _, row in items_df.iterrows()
    ]


@app.route("/")
def index():
    hero_options = get_hero_options()
    item_options = get_item_options()
    return render_template(
        "index.html", hero_options=hero_options, item_options=item_options
    )


@app.route("/predict", methods=["POST"])
def predict():
    data = request.form

    radiant_heroes = [int(data[f"r_hero_{i+1}"]) for i in range(5)]
    dire_heroes = [int(data[f"d_hero_{i+1}"]) for i in range(5)]
    radiant_items = [
        [int(data[f"r_{i+1}_item_{j}"]) for j in range(6)] for i in range(5)
    ]
    dire_items = [[int(data[f"d_{i+1}_item_{j}"]) for j in range(6)] for i in range(5)]

    feature_names = (
        [f"r_hero_{i+1}" for i in range(5)]
        + [f"d_hero_{i+1}" for i in range(5)]
        + [f"r_{i+1}_item_{j}" for i in range(5) for j in range(6)]
        + [f"d_{i+1}_item_{j}" for i in range(5) for j in range(6)]
    )

    input_data = (
        radiant_heroes
        + dire_heroes
        + [item for sublist in radiant_items for item in sublist]
        + [item for sublist in dire_items for item in sublist]
    )
    input_df = pd.DataFrame([input_data], columns=feature_names)

    dmatrix = xgb.DMatrix(input_df)
    prediction = model.predict(dmatrix)[0]

    if prediction == 1:
        result = "Team Radiant Menang"
    else:
        result = "Team Dire Menang"
        

    return render_template(
        "index.html",
        hero_options=get_hero_options(),
        item_options=get_item_options(),
        prediction=result,
        data=data
    )


if __name__ == "__main__":
    app.run(debug=True)
