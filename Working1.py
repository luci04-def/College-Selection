from flask import Flask, render_template, request
import pandas as pd
import re

PRED_CSV = r"C:/Users/Sukant/Downloads/tnea_cleaned.csv"

app = Flask(__name__)

def load_predictions(path=PRED_CSV):
    df = pd.read_csv(path, dtype=str)

    df = df.loc[:, ~df.columns.duplicated()]

    if "predicted_cutoff" in df.columns:
        df["predicted_cutoff"] = pd.to_numeric(df["predicted_cutoff"], errors="coerce")
    elif "cutoff" in df.columns:
        df["predicted_cutoff"] = pd.to_numeric(df["cutoff"], errors="coerce")
    else:
        raise ValueError("CSV must have 'predicted_cutoff' or 'cutoff' column")

    df = df.dropna(subset=["college_code", "college_name", "branch_code", "branch_name", "category", "predicted_cutoff"])

    df["college_code"] = df["college_code"].astype(str).str.strip()
    df["branch_code"] = df["branch_code"].astype(str).str.strip().str.upper()
    df["college_name"] = df["college_name"].astype(str).str.strip()

    df["branch_name"] = (
        df["branch_name"]
        .astype(str)
        .str.strip()
        .str.replace(r"^branch_name\s*", "", regex=True) 
        .str.replace(r"^[A-Z]{1,4}\s+branch_name\s*", "", regex=True) 
    )

    df = df[df["branch_name"].str.len() > 1]

    df["category"] = df["category"].astype(str).str.strip().str.upper()

    valid_categories = {"OC", "BC", "BCM", "MBC", "SC", "SCA", "ST"}
    df = df[df["category"].isin(valid_categories)]

    df = df.drop_duplicates(subset=["college_code", "branch_code", "category", "predicted_cutoff"])

    df["category_norm"] = df["category"]
    df["college_code_norm"] = df["college_code"]
    df["branch_code_norm"] = df["branch_code"]
    df["college_name_norm"] = df["college_name"].str.upper()
    df["branch_name_norm"] = df["branch_name"].str.upper()

    return df


def status_label(mark, cutoff):
    if mark >= cutoff:
        return "LIKELY"
    elif cutoff - mark <= 5:
        return "NEAR MISS"
    else:
        return "HARD REACH"

def check_selected_college(df, mark, college, branch, category):
    if not college:
        return None

    d = df[df["category_norm"] == category.strip().upper()]
    col_query = college.strip().upper()

    if re.fullmatch(r"\d{2,6}", col_query):
        d = d[d["college_code_norm"] == col_query]
    else:
        exact_match = d[d["college_name_norm"] == col_query]
        if not exact_match.empty:
            d = exact_match
        else:
            d = d[d["college_name_norm"].str.contains(col_query, na=False)]

    if branch:
        b = branch.strip().upper()
        branch_match = d[(d["branch_code_norm"] == b) | (d["branch_name_norm"].str.contains(b, na=False))]
        if not branch_match.empty:
            d = branch_match

    if d.empty:
        return None

    d["predicted_cutoff"] = pd.to_numeric(d["predicted_cutoff"], errors="coerce")
    d = d.dropna(subset=["predicted_cutoff"])

    row = d.iloc[0]
    cutoff = float(row["predicted_cutoff"])
    margin = round(mark - cutoff, 2)

    return {
        "college_name": row["college_name"],
        "college_code": row["college_code"],
        "branch_name": row["branch_name"],
        "branch_code": row["branch_code"],
        "predicted_cutoff": cutoff,
        "your_mark": mark,
        "margin": margin,
        "status": status_label(mark, cutoff),
        "category": category
    }


def recommend_colleges(df, mark, category, branch=None, top_n=10):
    d = df[df["category_norm"] == category.strip().upper()].copy()
    if branch:
        b = branch.strip().upper()
        d = d[(d["branch_code_norm"] == b) | (d["branch_name_norm"].str.contains(b, na=False))]

    if d.empty:
        return []

    d["predicted_cutoff"] = pd.to_numeric(d["predicted_cutoff"], errors="coerce")
    d = d.dropna(subset=["predicted_cutoff"])

    d["margin"] = mark - d["predicted_cutoff"]
    d = d[(d["margin"] >= 0) & (d["margin"] < 10)]

    if d.empty:
        return []

    d = d.sort_values("margin", ascending=False)

    recs = []
    for _, r in d.head(top_n).iterrows():
        recs.append({
            "college_name": r["college_name"],
            "college_code": r["college_code"],
            "branch_name": r["branch_name"],
            "branch_code": r["branch_code"],
            "predicted_cutoff": float(r["predicted_cutoff"]),
            "margin": round(r["margin"], 2),
            "status": "LIKELY"
        })
    return recs


def comparable_colleges(df, mark, category, branch=None, window=5):
    d = df[df["category_norm"] == category.strip().upper()].copy()
    if branch:
        b = branch.strip().upper()
        d = d[(d["branch_code_norm"] == b) | (d["branch_name_norm"].str.contains(b, na=False))]

    if d.empty:
        return []

    d["predicted_cutoff"] = pd.to_numeric(d["predicted_cutoff"], errors="coerce")
    d = d.dropna(subset=["predicted_cutoff"])

    d["margin"] = mark - d["predicted_cutoff"]
    close = d[d["margin"].between(-window, window)].sort_values("margin", ascending=False)

    recs = []
    for _, r in close.iterrows():
        recs.append({
            "college_name": r["college_name"],
            "college_code": r["college_code"],
            "branch_name": r["branch_name"],
            "branch_code": r["branch_code"],
            "predicted_cutoff": float(r["predicted_cutoff"]),
            "margin": round(r["margin"], 2),
            "status": status_label(mark, float(r["predicted_cutoff"]))
        })
    return recs


def safe_colleges(df, mark, category, branch=None, margin_safe=10):
    d = df[df["category_norm"] == category.strip().upper()].copy()
    if branch:
        b = branch.strip().upper()
        d = d[(d["branch_code_norm"] == b) | (d["branch_name_norm"].str.contains(b, na=False))]

    if d.empty:
        return []

    d["predicted_cutoff"] = pd.to_numeric(d["predicted_cutoff"], errors="coerce")
    d = d.dropna(subset=["predicted_cutoff"])

    d["margin"] = mark - d["predicted_cutoff"]
    d = d[d["margin"] >= margin_safe]

    if d.empty:
        return []

    d = d.sort_values("margin", ascending=False)

    recs = []
    for _, r in d.iterrows():
        recs.append({
            "college_name": r["college_name"],
            "college_code": r["college_code"],
            "branch_name": r["branch_name"],
            "branch_code": r["branch_code"],
            "predicted_cutoff": float(r["predicted_cutoff"]),
            "margin": round(r["margin"], 2),
            "status": "SAFE"
        })
    return recs


df = load_predictions()


@app.route("/", methods=["GET", "POST"])
def index():
    result, recs, comps, safe = None, [], [], []
    if request.method == "POST":
        mark = float(request.form.get("mark_percent"))
        category = request.form.get("category")
        college = request.form.get("college")
        branch = request.form.get("branch")

        result = check_selected_college(df, mark, college, branch, category)
        recs = recommend_colleges(df, mark, category, branch, top_n=10)
        comps = comparable_colleges(df, mark, category, branch, window=5)
        safe = safe_colleges(df, mark, category, branch, margin_safe=10)

    categories = sorted(df["category_norm"].dropna().unique().tolist())

    branches = sorted(((df["branch_name"] + " [" + df["branch_code"] + "]")
                      .dropna().unique().tolist()))

    colleges = sorted(df["college_name"].dropna().astype(str).unique().tolist())

    return render_template("index.html", categories=categories, branches=branches, colleges=colleges,
                           result=result, recs=recs, comps=comps, safe=safe)


if __name__ == "__main__":
    app.run(debug=True)
