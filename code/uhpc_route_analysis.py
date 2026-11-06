"""
Route-dependence analysis for:
  A. Tabani and R. Biswas, "Mixture or Method? A Quantitative Review of Route-Dependence
  in the Reported Performance of Ultra-High-Performance Concrete" (submitted).

Run from the repository root:   python code/uhpc_route_analysis.py
Reads data/ and coding/coding_sheet_v2.xlsx; writes tables and figures to results/.
  A. within-study contrasts (curing, composition, fibre, geometry, age)
  B. tensile test-method ratios and flexural size effect
  C. linear mixed-effects model (REML, random intercept for study) + sensitivity analyses
  D. reporting audit of the 64 experimental studies
Requires: see requirements.txt
"""
import argparse, os, re, time, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------------
# 1. Curing coding per study (from the 'Curing regime' column of the comparison sheet)
#    T = thermal >= 60 C (steam, hot water, oven, autoclave); A = ambient/moist
#    (water, fog, sealed, room); NR = not reported / unclear; NA = not experimental
#    MIXED = study tests more than one regime (row-level coding applies)
# ----------------------------------------------------------------------------------
STUDY_CURING = {
    1: ("MIXED", ""), 2: ("MIXED", "autoclave and steam are both T; contrast uses them separately"),
    3: ("MIXED", "no compressive rows in mechanical sheet"), 4: ("NR", ""), 5: ("NR", ""),
    6: ("T", ""), 7: ("A", "assumed: 'no thermal curing reported' - CHECK"), 8: ("MIXED", ""),
    9: ("T", "90 C water"), 10: ("T", ""), 11: ("A", ""), 12: ("A", ""), 13: ("A", ""),
    14: ("A", ""), 15: ("NR", ""), 16: ("A", ""), 17: ("A", ""), 18: ("A", ""), 19: ("A", ""),
    20: ("A", ""), 21: ("A", "fails UHPC screening - excluded"), 22: ("T", "90 C hot water"),
    23: ("A", ""), 24: ("T", ""), 25: ("A", "sealed 20 C"), 26: ("A", ""), 27: ("A", "fog room"),
    28: ("A", "assumed: 91 d, no heat reported - CHECK"),
    29: ("MIXED", "normal vs hot curing"),
    30: ("NR", "fog and thermal-activation series both exist; rows not labelled - CHECK"),
    31: ("A", ""), 32: ("NA", "review"), 33: ("A", ""), 34: ("A", "wet burlap then air"),
    35: ("MIXED", "oven ~150 C vs 28 d water (rows added in v17)"), 36: ("A", ""), 37: ("A", ""),
    38: ("A", ""), 39: ("A", ""), 40: ("A", ""), 41: ("MIXED", "steam 90 C 10 h vs water (Table 7 rows added in v17)"),
    42: ("NA", "review"), 43: ("A", ""), 44: ("A", ""), 45: ("T", "steam; T and duration NR"),
    46: ("NR", ""), 47: ("A", ""), 48: ("T", ""), 49: ("A", ""), 50: ("NR", ""), 51: ("T", ""),
    52: ("A", "fog room"), 53: ("T", ""), 54: ("A", ""), 55: ("A", ""), 56: ("A", ""),
    57: ("T", ""), 58: ("NR", "curing NR; 60 C seawater is exposure"),
    59: ("NR", "'standard-cured' only"), 60: ("A", ""), 61: ("T", ""), 62: ("NA", "numerical"),
    63: ("T", "80 C steam; 300-900 C rows are post-fire exposure"), 64: ("NR", "fire exposure study"),
    65: ("A", ""), 66: ("NR", "field blast test"), 67: ("T", ""), 68: ("T", "70 C steam"),
}
ROW_CURING_RULES = [  # applied only to MIXED studies, on 'Test condition'
    (r"steam|autoclave|thermal|hot|elevated|oven|90C|60C", "T"),
    (r"ambient|normal|sealed|RT|air|water", "A"),
]
STUDY8_BLANK = "A"   # Garas 2012 un-labelled row = ambient 23 C

EXCLUDE_STUDIES = {21: "fails UHPC screening", 62: "numerical study", 32: "review", 42: "review"}
EXCLUDE_MIX_PAT = r"range low|range high|population|18 fibre series|all five|NSC control|C-FT0"
EXCLUDE_COND_PAT = r"^(300C|450C|600C|900C)$"
BAD_VALUE_TYPES = {"equation-derived", "model-input"}

P_FC = "Compressive strength"
P_DT = "Direct tensile strength (peak)"
P_SP = "Splitting tensile strength"
P_MOR = "Flexural strength (MOR)"


def step(msg):
    tqdm.write(f"[{time.strftime('%H:%M:%S')}] {msg}")


def load(args):
    mech = pd.read_excel(args.mech, sheet_name="Mechanical data")
    mech["Row"] = mech.index + 2
    cum = pd.read_excel(args.cum, sheet_name=0)
    cum["Study_no"] = np.arange(1, len(cum) + 1)
    dur = pd.read_excel(args.dur, sheet_name="Durability data")
    durix = pd.read_excel(args.dur, sheet_name="Paper index")
    return mech, cum, dur, durix


# ----------------------------------------------------------------------------------
# 2. Default coding sheet
# ----------------------------------------------------------------------------------
def default_coding(mech, cum):
    sc = pd.DataFrame([{
        "Study_no": s, "Reference": cum.loc[s - 1, "Reference"] if s <= len(cum) else "",
        "Curing_text": str(cum.loc[s - 1, "Curing regime"])[:250] if s <= len(cum) else "",
        "Curing_class": c, "Coding_note": n} for s, (c, n) in STUDY_CURING.items()])

    comp = mech[mech.Property == P_FC].copy()
    inc, why, cur = [], [], []
    for _, r in comp.iterrows():
        reason = ""
        if r.Study_no in EXCLUDE_STUDIES: reason = EXCLUDE_STUDIES[r.Study_no]
        elif r.Value_type in BAD_VALUE_TYPES: reason = r.Value_type
        elif re.search(EXCLUDE_MIX_PAT, str(r["Mix / series"]), re.I): reason = "summary/range or non-UHPC row"
        elif re.search(EXCLUDE_COND_PAT, str(r["Test condition"])): reason = "post-fire exposure"
        inc.append("N" if reason else "Y"); why.append(reason)
        c = STUDY_CURING.get(r.Study_no, ("NR", ""))[0]
        if c == "MIXED":
            tc = str(r["Test condition"]); c = "NR"
            if r.Study_no == 8 and tc == "nan": c = STUDY8_BLANK
            for pat, lab in ROW_CURING_RULES:
                if re.search(pat, tc, re.I): c = lab; break
        cur.append(c)
    comp["Include"], comp["Exclude_reason"], comp["Row_curing"] = inc, why, cur
    # series label for Hung et al. 2020 (three un-named series of SS30/50/70 x 28/90 d)
    comp["Series"] = ""
    h = comp[comp.Study_no == 37].index
    for k, ix in enumerate(h): comp.loc[ix, "Series"] = "ABC"[k // 6]
    rows = comp[["Row", "Study_no", "Reference", "Mix / series", "Fibre_vol_pct", "Test condition",
                 "Age_d", "Value", "Value_type", "Include", "Exclude_reason", "Row_curing", "Series"]]
    contrasts = build_contrasts(rows)
    return sc, rows, contrasts


def build_contrasts(rows):
    r = rows[rows.Include == "Y"].copy()
    r["Age_k"] = r.Age_d.fillna(-1); r["TC_k"] = r["Test condition"].fillna("-")
    out, gid = [], [0]

    def grp(factor, s, by=None, mixes=None, drop=None, note=""):
        g0 = r[r.Study_no == s]
        if mixes is not None: g0 = g0[g0["Mix / series"].isin(mixes)]
        if drop is not None: g0 = g0[~g0["Mix / series"].isin(drop)]
        parts = [((), g0)] if by is None else list(g0.groupby(by))
        for _, g in parts:
            if len(g) < 2: continue
            gid[0] += 1
            for _, x in g.iterrows():
                out.append({"Contrast_id": gid[0], "Factor": factor, "Row": x.Row, "Study_no": s,
                            "Reference": x.Reference, "Mix / series": x["Mix / series"],
                            "Test condition": x["Test condition"], "Age_d": x.Age_d,
                            "Value": x.Value, "Row_curing": x.Row_curing, "Note": note})
    # curing - same mixture, different regime
    grp("Curing", 1); grp("Curing", 2, by="Mix / series", note="autoclave vs steam (both thermal)")
    grp("Curing", 8, note="7 d"); grp("Curing", 29, by="Mix / series", note="NC 91 d vs hot 2 d")
    grp("Curing", 35, by="Mix / series", note="oven vs 28 d water"); grp("Curing", 41, by="Mix / series")
    # specimen geometry - same mixture, different specimen
    grp("Geometry", 17, by="Mix / series"); grp("Geometry", 26)
    # composition - binder/filler changes at fixed curing, age, fibre
    grp("Composition", 2, by="TC_k"); grp("Composition", 14, drop=["Reference + 2.5%"])
    grp("Composition", 20); grp("Composition", 23); grp("Composition", 25)
    grp("Composition", 29, by="TC_k"); grp("Composition", 34, by="Age_k", note="CHECK variable")
    grp("Composition", 36, note="CHECK variable"); grp("Composition", 37, by=["Series", "Age_k"])
    grp("Composition", 44); grp("Composition", 47, by="Age_k"); grp("Composition", 54, drop=["Matrix"])
    grp("Composition", 56, by="Age_k", mixes=["UHPC-RGP", "UHPC-FA"])
    grp("Composition", 56, mixes=["M-RGP", "M-FA"]); grp("Composition", 59, by="Age_k")
    grp("Composition", 60, by="Age_k"); grp("Composition", 35, by="TC_k"); grp("Composition", 63, by="TC_k"); grp("Composition", 68)
    # fibre - content/type changes at fixed matrix
    grp("Fibre", 17, by="TC_k"); grp("Fibre", 22); grp("Fibre", 24); grp("Fibre", 33)
    grp("Fibre", 46); grp("Fibre", 48); grp("Fibre", 53, by="Age_k")
    grp("Fibre", 54, mixes=["Matrix", "R-0%"]); grp("Fibre", 41, by="TC_k", note="straight/hooked hybrid ratio at 2%")
    # age - context only
    grp("Age (context)", 27, mixes=["R-4"]); grp("Age (context)", 30, by="Mix / series", note="curing unclear")
    grp("Age (context)", 34, by="Mix / series"); grp("Age (context)", 37, by=["Series", "Mix / series"])
    grp("Age (context)", 47, by="Mix / series"); grp("Age (context)", 53, by="Mix / series")
    grp("Age (context)", 56, by="Mix / series", mixes=["UHPC-RGP", "UHPC-FA"])
    grp("Age (context)", 59, by="Mix / series")
    return pd.DataFrame(out)


# ----------------------------------------------------------------------------------
# 3. Analyses
# ----------------------------------------------------------------------------------
def within_study(contrasts):
    g = contrasts.groupby(["Contrast_id", "Factor", "Study_no", "Reference"])
    res = g.Value.agg(n="size", vmin="min", vmax="max").reset_index()
    res["Ratio_max_min"] = res.vmax / res.vmin
    res["Spread_pct"] = (res.Ratio_max_min - 1) * 100
    # one value per study and factor (median of its groups) so large studies do not dominate
    per_study = res.groupby(["Factor", "Study_no", "Reference"]).Ratio_max_min.median().reset_index()
    summ = per_study.groupby("Factor").Ratio_max_min.agg(
        studies="size", median="median", q1=lambda x: x.quantile(.25),
        q3=lambda x: x.quantile(.75), max="max").reset_index()
    return res, per_study, summ


def tensile_ratios(mech):
    t = mech[mech.Property.isin([P_DT, P_SP, P_MOR]) & ~mech.Value_type.isin(BAD_VALUE_TYPES)
             & ~mech.Study_no.isin(EXCLUDE_STUDIES)
             & ~mech["Test condition"].astype(str).str.contains("freeze", case=False)].copy()
    t["Age_k"] = t.Age_d.fillna(-1)
    key = ["Study_no", "Reference", "Mix / series", "Age_k"]
    pairs = []
    # each tensile test has its own specimen, so match on study + mixture + age only;
    # every numerator specimen (e.g. each flexural prism size) is paired with the denominator
    for num, den, lab in [(P_MOR, P_DT, "Flexural / direct"), (P_SP, P_DT, "Splitting / direct"),
                          (P_MOR, P_SP, "Flexural / splitting")]:
        a = t[t.Property == num][key + ["Test condition", "Value"]]
        b = t[t.Property == den].groupby(key).Value.mean().rename("Den").reset_index()
        x = a.merge(b, on=key)
        for _, r in x.iterrows():
            pairs.append({"Ratio": lab, "Study_no": r.Study_no, "Reference": r.Reference,
                          "Mix / series": r["Mix / series"], "Age_d": r.Age_k,
                          "Numerator specimen": r["Test condition"], "Numerator": r.Value,
                          "Denominator": r.Den, "Value": r.Value / r.Den})
    p = pd.DataFrame(pairs)
    s = p.groupby("Ratio").agg(pairs=("Value", "size"), studies=("Study_no", "nunique"),
                               median=("Value", "median"), min=("Value", "min"),
                               max=("Value", "max")).reset_index() if len(p) else p
    # flexural specimen-size effect: same mixture, different prism size
    f = t[(t.Property == P_MOR) & t["Test condition"].notna()]
    g = f.groupby(key).agg(sizes=("Test condition", "nunique"), vmax=("Value", "max"),
                           vmin=("Value", "min")).reset_index()
    g = g[g.sizes > 1]
    g["Ratio_max_min"] = g.vmax / g.vmin
    return p, s, g


def pooled_model(rows):
    d = rows[(rows.Include == "Y") & rows.Row_curing.isin(["T", "A", "NR"])].copy()
    d["Age_d"] = pd.to_numeric(d.Age_d, errors="coerce")
    d["lnfc"] = np.log(d.Value.astype(float))
    d["cure"] = pd.Categorical(d.Row_curing, categories=["A", "T", "NR"])
    d["age"] = pd.cut(d.Age_d, [0, 7, 27, 28, 90, 1e5], labels=["<=7", "8-27", "28", "29-90", ">90"]
                      ).astype(object).fillna("NR").astype(str)
    d["vf"] = pd.cut(d.Fibre_vol_pct.astype(float), [-0.01, 0, 1, 2, 10], labels=["0", "0-1", "1-2", ">2"]
                     ).astype(object).fillna("NR").astype(str)
    d["study"] = d.Study_no.astype(str)
    out = {"n_rows": len(d), "n_studies": d.study.nunique(),
           "rows_by_curing": d.Row_curing.value_counts().to_dict()}
    # OLS ladder
    for name, f in [("R2_curing_only", "lnfc ~ C(cure)"),
                    ("R2_curing_age_vf", "lnfc ~ C(cure) + C(age) + C(vf)"),
                    ("R2_plus_study_dummies", "lnfc ~ C(cure) + C(age) + C(vf) + C(study)")]:
        out[name] = smf.ols(f, d).fit().rsquared
    # mixed model
    m = smf.mixedlm("lnfc ~ C(cure) + C(age) + C(vf)", d, groups=d["study"]).fit(reml=True)
    b, ci = m.params, m.conf_int()
    k = "C(cure)[T.T]"
    out["Thermal_vs_ambient_ratio"] = float(np.exp(b[k]))
    out["Thermal_vs_ambient_CI95"] = (float(np.exp(ci.loc[k, 0])), float(np.exp(ci.loc[k, 1])))
    out["Thermal_p"] = float(m.pvalues[k])
    X = m.model.exog; fe = X @ m.fe_params.values
    vf_, vs, ve = np.var(fe), float(m.cov_re.iloc[0, 0]), float(m.scale)
    out["marginal_R2"] = vf_ / (vf_ + vs + ve)
    out["conditional_R2"] = (vf_ + vs) / (vf_ + vs + ve)
    out["between_study_share_of_random_var"] = vs / (vs + ve)
    return out, m.summary().as_text(), d


def audit(cum, mech, dur, durix, sc):
    a = []
    for s in range(1, len(cum) + 1):
        if s in (21, 32, 42, 62): continue   # fails screening, reviews, numerical
        c = cum.loc[s - 1]
        txt = " ".join(str(c[k]) for k in ["Curing regime", "Testing days", "Mechanical properties"])
        mrows = mech[mech.Study_no == s]
        cl = sc.loc[sc.Study_no == s, "Curing_class"]
        cl = cl.iloc[0] if len(cl) else "NR"
        td = str(c["Testing days"])
        drows = dur[dur.Study_no == s]
        nd = durix.loc[durix.Study_no == s, "Records"]
        a.append({
            "Study_no": s, "Reference": c.Reference,
            "Curing stated": cl not in ("NR", "NA"),
            "Test age stated": bool(mrows.Age_d.notna().any()) or not re.search(r"\bNR\b|not reported", td, re.I),
            "Fibre Vf stated": bool(mrows.Fibre_vol_pct.notna().any()) or bool(
                re.search(r"vol|%|no fibre|without fibre", str(c["Fibre type and quantity"]), re.I)),
            "Specimen geometry stated (text search)": bool(re.search(r"cube|cylind|prism|dog.?bone|\d+\s*[x×]\s*\d+", txt, re.I)),
            "Any numeric durability": bool(len(nd) and float(nd.iloc[0]) > 0),
            "Durability under load / pre-cracked": bool(drows["Test condition"].astype(str).str.contains(
                r"load|crack|pre-?tension", case=False).any() | drows["Notes"].astype(str).str.contains(
                r"under load|pre-?crack|pre-?tension|sustained load", case=False).any()),
        })
    a = pd.DataFrame(a)
    s = a.drop(columns=["Study_no", "Reference"]).mean().mul(100).round(1).rename("% of studies").reset_index()
    s.columns = ["Reporting item", "% of studies"]
    s["n studies"] = len(a)
    return a, s



def model_details(d):
    """Full fixed-effects table, variance components and sensitivity fits (Table A2)."""
    f = "lnfc ~ C(cure) + C(age) + C(vf)"; k = "C(cure)[T.T]"
    fit = lambda x, reml=True: smf.mixedlm(f, x, groups=x["study"]).fit(reml=reml)
    m = fit(d); ci = m.conf_int().loc[m.fe_params.index]
    fe = pd.DataFrame({"coef": m.fe_params, "se": m.bse_fe, "p": m.pvalues[m.fe_params.index],
                       "ratio": np.exp(m.fe_params), "ratio_lo": np.exp(ci[0]), "ratio_hi": np.exp(ci[1])}).reset_index()
    vs, ve = float(m.cov_re.iloc[0, 0]), float(m.scale)
    var = pd.DataFrame([{"item": "between-study variance", "value": vs}, {"item": "residual variance", "value": ve},
                        {"item": "ICC", "value": vs / (vs + ve)}, {"item": "n obs", "value": len(d)},
                        {"item": "n studies", "value": d.study.nunique()}])
    sens = []
    def add(name, x, reml=True):
        mm = fit(x, reml); c = np.exp(mm.conf_int().loc[k].values)
        sens.append({"analysis": name, "n_obs": len(x), "n_studies": x.study.nunique(),
                     "thermal_ratio": float(np.exp(mm.params[k])), "lo": c[0], "hi": c[1]})
    add("main (REML)", d); add("maximum likelihood", d, reml=False)
    for s in tqdm(d.study.unique(), desc="leave-one-study-out", leave=False):
        add(f"without study {s}", d[d.study != s])
    sm = d.groupby("study").Value.mean(); lo, hi = sm.quantile([.1, .9])
    add("exclude top/bottom 10% study-mean strength", d[d.study.isin(sm[(sm >= lo) & (sm <= hi)].index)])
    if "Value_type" in d: add("measured values only", d[d.Value_type == "measured"])
    z = m.resid / m.resid.std(); add("exclude |std resid| > 3", d[(z.abs() <= 3).values])
    return fe, var, pd.DataFrame(sens)

# ----------------------------------------------------------------------------------
# 4. Figures
# ----------------------------------------------------------------------------------
def fig_within(per_study, path):
    order = ["Curing", "Composition", "Fibre", "Geometry", "Age (context)"]
    cols = {"Curing": "#C0392B", "Composition": "#2E86C1", "Fibre": "#7D3C98",
            "Geometry": "#D68910", "Age (context)": "#7F8C8D"}
    fig, ax = plt.subplots(figsize=(7, 4.2))
    rng = np.random.default_rng(1)
    for i, f in enumerate(order):
        v = per_study.loc[per_study.Factor == f, "Ratio_max_min"].values
        if not len(v): continue
        ax.scatter(i + rng.uniform(-.12, .12, len(v)), (v - 1) * 100, s=28, color=cols[f],
                   alpha=.8, edgecolor="white", lw=.5, zorder=3)
        ax.hlines(np.median((v - 1) * 100), i - .28, i + .28, color="black", lw=2, zorder=4)
        ax.text(i, -6, f"n={len(v)}", ha="center", va="top", fontsize=8, color="#555")
    ax.set_xticks(range(len(order))); ax.set_xticklabels(order)
    ax.set_ylabel("Within-study spread in compressive strength (%)\n(max/min − 1, one point per study)")
    ax.set_ylim(-12, None); ax.grid(axis="y", color="#ddd", lw=.6); ax.set_axisbelow(True)
    for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
    fig.tight_layout(); fig.savefig(path, dpi=300); plt.close(fig)


def fig_tensile(p, path):
    if not len(p): return
    order = [x for x in ["Flexural / direct", "Splitting / direct", "Flexural / splitting"] if x in set(p.Ratio)]
    fig, ax = plt.subplots(figsize=(6, 4))
    rng = np.random.default_rng(2)
    for i, f in enumerate(order):
        v = p.loc[p.Ratio == f, "Value"].values
        ax.scatter(i + rng.uniform(-.12, .12, len(v)), v, s=26, color="#2E86C1", alpha=.8,
                   edgecolor="white", lw=.5, zorder=3)
        ax.hlines(np.median(v), i - .28, i + .28, color="black", lw=2, zorder=4)
        ax.text(i, ax.get_ylim()[0], f"n={len(v)}", ha="center", va="bottom", fontsize=8, color="#555")
    ax.axhline(1, color="#999", ls="--", lw=1)
    ax.set_xticks(range(len(order))); ax.set_xticklabels(order)
    ax.set_ylabel("Ratio on the same mixture")
    ax.grid(axis="y", color="#ddd", lw=.6); ax.set_axisbelow(True)
    for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
    fig.tight_layout(); fig.savefig(path, dpi=300); plt.close(fig)


# ----------------------------------------------------------------------------------
def main():
    try:
        ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repo root (script lives in code/)
    except NameError:                                                        # running inside Jupyter
        ROOT = os.path.dirname(os.getcwd()) if os.path.basename(os.getcwd()) == "code" else os.getcwd()
    BASE = os.path.join(ROOT, "data")
    ap = argparse.ArgumentParser()
    import glob

    def find(pat):  # newest file in BASE whose name matches the pattern
        hits = [f for f in glob.glob(os.path.join(BASE, pat)) if not os.path.basename(f).startswith("~$")]
        if not hits:
            raise FileNotFoundError(f"No file matching '{pat}' in {BASE}. Files there: {os.listdir(BASE)}")
        return max(hits, key=os.path.getmtime)

    ap.add_argument("--mech", default=None)
    ap.add_argument("--cum", default=None)
    ap.add_argument("--dur", default=None)
    ap.add_argument("--coding", default=os.path.join(ROOT, "coding", "coding_sheet_v2.xlsx"))
    ap.add_argument("--out", default=os.path.join(ROOT, "results"))
    a, _ = ap.parse_known_args()   # works in Jupyter too
    a.mech = a.mech or find("*mechanical_data*.xlsx")
    a.cum = a.cum or find("*cumulative_comparison*.xlsx")
    a.dur = a.dur or find("*durability_data*.xlsx")
    for k in ("mech", "cum", "dur", "coding"):
        print(f"{k:7s}: {getattr(a, k)}")
    os.makedirs(a.out, exist_ok=True)
    stages = ["load", "coding", "within-study", "tensile", "mixed model", "audit", "figures", "write"]
    bar = tqdm(total=len(stages), desc="Analysis", unit="stage",
               bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")

    mech, cum, dur, durix = load(a); bar.update()

    checked = a.coding
    if os.path.exists(checked):
        step(f"using YOUR checked coding sheet: {checked}")
        sc = pd.read_excel(checked, sheet_name="study_curing")
        rows = pd.read_excel(checked, sheet_name="compression_rows")
        contrasts = pd.read_excel(checked, sheet_name="contrast_groups")
        keep = set(rows.loc[rows.Include.astype(str).str.upper() == "Y", "Row"])
        contrasts = contrasts[contrasts.Row.isin(keep)]   # rows you set to N drop out of contrasts too
    else:
        sc, rows, contrasts = default_coding(mech, cum)
        with pd.ExcelWriter(os.path.join(a.out, "coding_sheet_v2.xlsx")) as w:
            sc.to_excel(w, sheet_name="study_curing", index=False)
            rows.to_excel(w, sheet_name="compression_rows", index=False)
            contrasts.to_excel(w, sheet_name="contrast_groups", index=False)
        step("wrote coding_sheet_v1.xlsx - check it, save as coding_sheet_checked.xlsx, re-run")
    bar.update()

    grp, per_study, summ = within_study(contrasts); bar.update()
    tp, ts, fsz = tensile_ratios(mech); bar.update()
    mres, msum, mdata = pooled_model(rows); fe, var, sens = model_details(mdata); bar.update()
    aud, auds = audit(cum, mech, dur, durix, sc); bar.update()
    fig_within(per_study, os.path.join(a.out, "fig_within_study_spread.png"))
    fig_tensile(tp, os.path.join(a.out, "fig_tensile_test_ratios.png")); bar.update()

    with pd.ExcelWriter(os.path.join(a.out, "route_analysis_results_v3.xlsx")) as w:
        summ.to_excel(w, sheet_name="A_factor_summary", index=False)
        per_study.to_excel(w, sheet_name="A_per_study", index=False)
        grp.to_excel(w, sheet_name="A_contrast_groups", index=False)
        ts.to_excel(w, sheet_name="B_tensile_summary", index=False)
        tp.to_excel(w, sheet_name="B_tensile_pairs", index=False)
        fsz.to_excel(w, sheet_name="B_flexural_size_effect", index=False)
        pd.DataFrame([{"Item": k, "Value": str(v)} for k, v in mres.items()]).to_excel(w, sheet_name="C_mixed_model", index=False)
        fe.to_excel(w, sheet_name="C_fixed_effects", index=False)
        var.to_excel(w, sheet_name="C_variance", index=False)
        sens.to_excel(w, sheet_name="C_sensitivity", index=False)
        auds.to_excel(w, sheet_name="D_audit_summary", index=False)
        aud.to_excel(w, sheet_name="D_audit_per_study", index=False)
    with open(os.path.join(a.out, "mixed_model_summary.txt"), "w") as f:
        f.write(msum + "\n\n" + "\n".join(f"{k}: {v}" for k, v in mres.items()))
    bar.update(); bar.close()

    print("\n=== A. Within-study spread (one value per study) ===\n", summ.round(3).to_string(index=False))
    print("\n=== B. Tensile test-method ratios ===\n", ts.round(3).to_string(index=False) if len(ts) else "none")
    print("\n=== B2. Flexural size effect (same mixture) ===\n", fsz.round(3).to_string(index=False) if len(fsz) else "none")
    print("\n=== C. Pooled mixed model ===")
    for k, v in mres.items(): print(f"  {k}: {v}")
    loo = sens[sens.analysis.str.startswith("without study")].thermal_ratio
    print(f"\n=== C2. Sensitivity: leave-one-out {loo.min():.3f}-{loo.max():.3f} ===\n", sens[~sens.analysis.str.startswith("without study")].round(3).to_string(index=False))
    print("\n=== D. Reporting audit ===\n", auds.to_string(index=False))


if __name__ == "__main__":
    main()
