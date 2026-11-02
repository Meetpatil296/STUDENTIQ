"""
app.py - STUDENTIQ: AI student performance analytics dashboard.
Run:  streamlit run app.py
All predictions come from the saved Random Forest pipeline
(models/random_forest_model.pkl); nothing is hardcoded.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from common import (CLASS_ORDER, FEATURES, LABELS, NUMERIC_FEATURES, TARGET,
                    clean_dataset, explain_prediction, load_dataset,
                    load_metrics, load_model, predict_student, validate_input)

st.set_page_config(page_title="STUDENTIQ", page_icon="🎓", layout="wide")

COLORS = {"Poor": "#f87171", "Average": "#fbbf24", "Good": "#34d399"}
BLUE, CYAN, INDIGO = "#3b82f6", "#22d3ee", "#818cf8"
GRID = "rgba(148,163,184,.14)"

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stApp {font-family: 'Inter', sans-serif;}
.stApp {
  background:
    radial-gradient(900px 500px at 8% -5%, rgba(34,211,238,.13), transparent 60%),
    radial-gradient(800px 500px at 95% 5%, rgba(59,130,246,.16), transparent 60%),
    linear-gradient(160deg, #070d1d 0%, #0a1124 45%, #111827 100%);
}
#MainMenu, footer {visibility: hidden;}
header[data-testid="stHeader"] {background: transparent;}
.block-container {padding-top: 2rem; max-width: 1280px;}

/* ---------- sidebar ---------- */
section[data-testid="stSidebar"] {
  background: rgba(10,17,36,.72); backdrop-filter: blur(18px);
  border-right: 1px solid rgba(148,163,184,.14);
}
.brand {display:flex; align-items:center; gap:.7rem; padding:.4rem .3rem 1.2rem;}
.brand-logo {width:42px; height:42px; border-radius:12px; display:flex; align-items:center;
  justify-content:center; font-size:1.3rem;
  background: linear-gradient(135deg, #2563eb, #22d3ee);
  box-shadow: 0 6px 24px rgba(34,211,238,.35);}
.brand-name {font-weight:800; font-size:1.25rem; letter-spacing:.04em;
  background: linear-gradient(90deg,#fff,#7dd3fc); -webkit-background-clip:text;
  -webkit-text-fill-color: transparent;}
.brand-sub {font-size:.72rem; color:#94a3b8; margin-top:-2px;}
section[data-testid="stSidebar"] div[role="radiogroup"] {gap:.3rem;}
section[data-testid="stSidebar"] div[role="radiogroup"] label {
  padding:.65rem .9rem; border-radius:12px; border:1px solid transparent;
  transition: all .2s ease; width:100%; cursor:pointer;
}
section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
  background: rgba(59,130,246,.14); border-color: rgba(34,211,238,.3);
  transform: translateX(3px);
}
section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
  background: linear-gradient(90deg, rgba(37,99,235,.35), rgba(34,211,238,.18));
  border-color: rgba(34,211,238,.45);
}
section[data-testid="stSidebar"] div[role="radiogroup"] label > div:first-child {display:none;}
.side-card {margin-top:1.4rem; padding:.9rem 1rem; border-radius:14px; font-size:.78rem;
  color:#94a3b8; background: rgba(255,255,255,.04); border:1px solid rgba(148,163,184,.14);}
.side-card b {color:#e2e8f0;}

/* ---------- glass + cards ---------- */
.glass, div[data-testid="stPlotlyChart"], div[data-testid="stDataFrame"] {
  background: rgba(255,255,255,.045); backdrop-filter: blur(16px);
  border: 1px solid rgba(148,163,184,.16); border-radius: 18px;
  box-shadow: 0 10px 40px rgba(2,6,23,.45);
  transition: transform .25s ease, border-color .25s ease, box-shadow .25s ease;
}
div[data-testid="stPlotlyChart"] {padding: .4rem;}
.glass:hover, div[data-testid="stPlotlyChart"]:hover {
  transform: translateY(-3px); border-color: rgba(34,211,238,.45);
  box-shadow: 0 16px 48px rgba(34,211,238,.12);
}
.hero {padding:1.8rem 2rem; border-radius:22px; margin-bottom:1.4rem;
  background: linear-gradient(120deg, rgba(37,99,235,.35), rgba(34,211,238,.12) 60%, rgba(255,255,255,.03));
  border:1px solid rgba(34,211,238,.28); backdrop-filter: blur(16px);}
.hero h1 {margin:0; font-size:2rem; font-weight:800; color:#fff;}
.hero p {margin:.4rem 0 0; color:#a5b4c8; font-size:.98rem;}
.kpi {padding:1.1rem 1.2rem; height:100%;}
.kpi-icon {font-size:1.25rem; width:38px; height:38px; border-radius:11px; display:flex;
  align-items:center; justify-content:center; margin-bottom:.7rem;
  background: linear-gradient(135deg, rgba(37,99,235,.55), rgba(34,211,238,.35));}
.kpi-label {font-size:.78rem; color:#94a3b8; font-weight:500;}
.kpi-value {font-size:1.85rem; font-weight:800; color:#fff; line-height:1.15;}
.kpi-sub {font-size:.74rem; color:#67e8f9; margin-top:.15rem;}
.sec-title {font-size:1.05rem; font-weight:700; color:#e2e8f0; margin:1.4rem 0 .6rem;}
.result {padding:1.3rem 1.5rem; border-radius:18px; border-left:6px solid var(--c);
  background: linear-gradient(120deg, rgba(255,255,255,.07), rgba(255,255,255,.02));
  border-top:1px solid rgba(148,163,184,.16); border-right:1px solid rgba(148,163,184,.16);
  border-bottom:1px solid rgba(148,163,184,.16); backdrop-filter: blur(16px);}
.result small {color:#94a3b8;}
.result h2 {margin:.1rem 0; font-size:2.2rem; font-weight:800; color:var(--c);}
.chip {display:inline-block; padding:.28rem .7rem; margin:.2rem .25rem .2rem 0; border-radius:999px;
  font-size:.78rem; border:1px solid rgba(148,163,184,.25); background: rgba(255,255,255,.05);}
.chip.good {border-color: rgba(52,211,153,.5); color:#6ee7b7;}
.chip.warn {border-color: rgba(251,191,36,.5); color:#fcd34d;}
.advice {padding:1rem 1.2rem; border-radius:14px; margin-top:.8rem; font-size:.92rem;
  background: rgba(34,211,238,.07); border:1px solid rgba(34,211,238,.25); color:#cffafe;}
.placeholder {padding:3rem 1.5rem; text-align:center; color:#94a3b8; border-radius:18px;
  border:1px dashed rgba(148,163,184,.3);}

/* ---------- inputs ---------- */
.stButton > button, div[data-testid="stFormSubmitButton"] > button {
  width:100%; border:0; border-radius:12px; padding:.7rem 1rem; font-weight:700; color:#fff;
  background: linear-gradient(90deg, #2563eb, #22d3ee);
  box-shadow: 0 8px 26px rgba(34,211,238,.28); transition: all .2s ease;
}
.stButton > button:hover, div[data-testid="stFormSubmitButton"] > button:hover {
  transform: translateY(-2px); box-shadow: 0 12px 32px rgba(34,211,238,.45); color:#fff;
}
div[data-testid="stForm"] {background: rgba(255,255,255,.04); backdrop-filter: blur(16px);
  border:1px solid rgba(148,163,184,.16); border-radius:18px; padding:1.2rem;}
button[data-baseweb="tab"] {font-weight:600;}
@media (max-width: 768px) {
  .hero h1 {font-size:1.4rem;} .kpi-value {font-size:1.4rem;} .block-container {padding-top:1rem;}
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ----------------------------------------------------------------------
# Cached loaders (same data sources as before)
# ----------------------------------------------------------------------
@st.cache_data
def get_raw():
    return load_dataset()


@st.cache_data
def get_clean():
    return clean_dataset(load_dataset())


@st.cache_data
def get_metrics():
    return load_metrics()


@st.cache_resource
def get_model():
    return load_model()


# ----------------------------------------------------------------------
# UI helpers
# ----------------------------------------------------------------------
def hex_rgba(h, a):
    h = h.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{a})"


def style(fig, height=360):
    fig.update_layout(
        template="plotly_dark", height=height,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#cbd5e1"),
        margin=dict(l=14, r=14, t=44, b=14),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor="#0f172a", font_size=13),
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


def show(fig):
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})


def hero(title, subtitle):
    st.markdown(f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p></div>', unsafe_allow_html=True)


def section(text):
    st.markdown(f'<div class="sec-title">{text}</div>', unsafe_allow_html=True)


def kpi(label, value, sub="", icon="📊"):
    return (f'<div class="kpi glass"><div class="kpi-icon">{icon}</div>'
            f'<div class="kpi-label">{label}</div><div class="kpi-value">{value}</div>'
            f'<div class="kpi-sub">{sub}</div></div>')


def kpi_row(items):
    cols = st.columns(len(items))
    for col, item in zip(cols, items):
        col.markdown(kpi(*item), unsafe_allow_html=True)


def norm(df, col, v):
    lo, hi = df[col].min(), df[col].max()
    return float(np.clip((v - lo) / (hi - lo) * 100, 0, 100)) if hi > lo else 50.0


def radar(df, series):
    """series: {name: (list_of_values_per_NUMERIC_FEATURE_already_0_100, colour)}"""
    cats = [LABELS[c] for c in NUMERIC_FEATURES]
    fig = go.Figure()
    for name, (vals, color) in series.items():
        fig.add_trace(go.Scatterpolar(
            r=list(vals) + [vals[0]], theta=cats + [cats[0]], name=name, fill="toself",
            line=dict(color=color, width=2), fillcolor=hex_rgba(color, .16)))
    fig.update_layout(polar=dict(
        bgcolor="rgba(0,0,0,0)",
        radialaxis=dict(range=[0, 100], gridcolor=GRID, showticklabels=False),
        angularaxis=dict(gridcolor=GRID)))
    return style(fig, 400)


# ----------------------------------------------------------------------
# Pages
# ----------------------------------------------------------------------
def page_overview():
    df, m = get_clean(), get_metrics()
    hero("Student performance intelligence",
         "Random Forest analytics that flag Good, Average and Poor performers early.")
    kpi_row([
        ("Students analysed", f"{len(df):,}", f"{m['dataset']['raw_rows']:,} raw records", "🎓"),
        ("Model accuracy", f"{m['accuracy'] * 100:.1f}%", "Unseen 20% test set", "🎯"),
        ("Macro F1-score", f"{m['f1_macro'] * 100:.1f}%", "Balanced across classes", "⚡"),
        ("5-fold CV accuracy", f"{m['cv_accuracy_mean'] * 100:.1f}%", f"± {m['cv_accuracy_std'] * 100:.1f}", "🧪"),
    ])

    c1, c2 = st.columns([1, 1.2])
    with c1:
        section("Performance mix")
        counts = df[TARGET].value_counts().reindex(CLASS_ORDER)
        fig = go.Figure(go.Pie(labels=counts.index, values=counts.values, hole=.62,
                               marker=dict(colors=[COLORS[c] for c in counts.index],
                                           line=dict(color="#0a1124", width=3)),
                               textinfo="percent"))
        fig.add_annotation(text=f"<b>{len(df):,}</b><br>students", showarrow=False,
                           font=dict(size=16, color="#fff"))
        show(style(fig, 360))
    with c2:
        section("What drives performance")
        imp = pd.Series(m["feature_importance"]).sort_values()
        fig = go.Figure(go.Bar(x=imp.values, y=[i.replace("_", " ") for i in imp.index],
                               orientation="h",
                               marker=dict(color=imp.values,
                                           colorscale=[[0, "#1d4ed8"], [1, CYAN]])))
        fig.update_layout(xaxis_title="Importance")
        show(style(fig, 360))

    section("Average student profile by category")
    means = df.groupby(TARGET)[NUMERIC_FEATURES].mean()
    series = {cls: ([norm(df, c, means.loc[cls, c]) for c in NUMERIC_FEATURES], COLORS[cls])
              for cls in CLASS_ORDER if cls in means.index}
    show(radar(df, series))
    st.caption("Each axis is scaled 0-100 between the lowest and highest value in the dataset.")


def page_prediction():
    df, model, m = get_clean(), get_model(), get_metrics()
    hero("Predict student performance",
         "Enter a student's profile and the trained Random Forest returns a category with confidence.")
    left, right = st.columns([1, 1.25], gap="large")

    with left:
        with st.form("prediction_form"):
            study = st.slider("Study hours per day", 0.0, 14.0, 3.0, 0.5)
            attendance = st.slider("Attendance (%)", 0.0, 100.0, 75.0, 1.0)
            previous = st.slider("Previous exam marks (0-100)", 0.0, 100.0, 60.0, 1.0)
            assignment = st.slider("Assignment scores (0-100)", 0.0, 100.0, 65.0, 1.0)
            internal = st.slider("Internal assessment marks (0-100)", 0.0, 100.0, 60.0, 1.0)
            sleep = st.slider("Sleep hours per day", 2.0, 14.0, 7.0, 0.5)
            a, b = st.columns(2)
            extra = a.selectbox("Extracurricular activity", ["Yes", "No"], index=1)
            internet = b.selectbox("Internet availability", ["Yes", "No"], index=0)
            submitted = st.form_submit_button("Predict performance")

    with right:
        if not submitted:
            st.markdown('<div class="placeholder">🔮<br><br>Set the student details and press '
                        '<b>Predict performance</b> to see the analysis.</div>', unsafe_allow_html=True)
            return
        values = {
            "Study_Hours_Per_Day": study, "Attendance_Percentage": attendance,
            "Previous_Exam_Marks": previous, "Assignment_Scores": assignment,
            "Internal_Assessment_Marks": internal, "Sleep_Hours": sleep,
            "Extracurricular_Activity": extra, "Internet_Availability": internet,
        }
        errors = validate_input(values)
        if errors:
            for e in errors:
                st.error(e)
            return
        try:
            label, probs = predict_student(model, values)
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
            return

        color = COLORS.get(label, BLUE)
        st.markdown(
            f'<div class="result" style="--c:{color}"><small>Predicted performance</small>'
            f'<h2>{label}</h2><small>Model confidence: <b style="color:#fff">'
            f'{probs[label] * 100:.1f}%</b></small></div>', unsafe_allow_html=True)

        g1, g2 = st.columns(2)
        with g1:
            gauge = go.Figure(go.Indicator(
                mode="gauge+number", value=probs[label] * 100,
                number=dict(suffix="%", font=dict(color="#fff")),
                gauge=dict(axis=dict(range=[0, 100], tickcolor="#64748b"),
                           bar=dict(color=color), bgcolor="rgba(255,255,255,.04)", borderwidth=0)))
            gauge.update_layout(title="Confidence")
            show(style(gauge, 260))
        with g2:
            p = pd.Series(probs).reindex(CLASS_ORDER)
            bars = go.Figure(go.Bar(x=p.values * 100, y=p.index, orientation="h",
                                    marker=dict(color=[COLORS[c] for c in p.index]),
                                    text=[f"{v * 100:.1f}%" for v in p.values], textposition="auto"))
            bars.update_layout(title="Probability by category", xaxis=dict(range=[0, 100]))
            show(style(bars, 260))

        section("Student vs typical student in this category")
        peer = df[df[TARGET] == label][NUMERIC_FEATURES].mean()
        show(radar(df, {
            "This student": ([norm(df, c, values[c]) for c in NUMERIC_FEATURES], CYAN),
            f"Average {label}": ([norm(df, c, peer[c]) for c in NUMERIC_FEATURES], INDIGO),
        }))

        info = explain_prediction(values, df, m["feature_importance"])
        section("Why this prediction?")
        chips = "".join(f'<span class="chip good">✅ {s}</span>' for s in info["strengths"])
        chips += "".join(f'<span class="chip warn">⚠️ {s}</span>' for s in info["concerns"])
        if not chips:
            chips = '<span class="chip">All inputs are close to the typical student.</span>'
        st.markdown(chips, unsafe_allow_html=True)
        top = ", ".join(k.replace("_", " ") for k, _ in info["top_features"])
        st.caption(f"The model relies most on: {top}.")

        advice = {
            "Poor": "This student may benefit from extra academic support, such as mentoring or remedial classes.",
            "Good": "This student is likely to perform well. Keep up the current study habits.",
            "Average": "This student is likely to perform averagely. Improving the weaker areas can raise the result.",
        }[label]
        st.markdown(f'<div class="advice">💡 {advice}</div>', unsafe_allow_html=True)
        st.caption("The prediction comes from the trained Random Forest model and is an indication, not a final judgement.")


def page_analytics():
    df = get_clean()
    hero("Advanced analytics", "Explore how study habits and marks relate to performance.")
    t1, t2, t3, t4 = st.tabs(["Distributions", "Relationships", "Correlations", "Segments"])

    with t1:
        col = st.selectbox("Feature", NUMERIC_FEATURES, format_func=lambda c: LABELS[c])
        a, b = st.columns(2)
        with a:
            fig = px.violin(df, x=TARGET, y=col, color=TARGET, box=True, points=False,
                            category_orders={TARGET: CLASS_ORDER}, color_discrete_map=COLORS,
                            labels={col: LABELS[col]})
            fig.update_layout(showlegend=False, title=f"{LABELS[col]} by category")
            show(style(fig, 400))
        with b:
            fig = px.histogram(df, x=col, color=TARGET, barmode="overlay", opacity=.65, nbins=30,
                               category_orders={TARGET: CLASS_ORDER}, color_discrete_map=COLORS,
                               labels={col: LABELS[col]})
            fig.update_layout(title=f"Distribution of {LABELS[col]}")
            show(style(fig, 400))

    with t2:
        c1, c2 = st.columns(2)
        x = c1.selectbox("X axis", NUMERIC_FEATURES, index=2, format_func=lambda c: LABELS[c])
        y = c2.selectbox("Y axis", NUMERIC_FEATURES, index=1, format_func=lambda c: LABELS[c])
        fig = px.scatter(df, x=x, y=y, color=TARGET, opacity=.7,
                         category_orders={TARGET: CLASS_ORDER}, color_discrete_map=COLORS,
                         labels={x: LABELS[x], y: LABELS[y]}, hover_data=["Student_ID"])
        fig.update_layout(title=f"{LABELS[y]} vs {LABELS[x]}")
        show(style(fig, 480))

    with t3:
        corr = df[NUMERIC_FEATURES].copy()
        corr["Performance level"] = df[TARGET].map({c: i for i, c in enumerate(CLASS_ORDER)})
        cm = corr.corr()
        names = [LABELS.get(c, c) for c in cm.columns]
        fig = px.imshow(cm.values, x=names, y=names, text_auto=".2f", zmin=-1, zmax=1,
                        color_continuous_scale=[[0, "#f87171"], [.5, "#0f172a"], [1, CYAN]])
        fig.update_layout(title="Correlation heatmap (Poor=0, Average=1, Good=2)")
        show(style(fig, 560))

    with t4:
        c1, c2 = st.columns(2)
        for holder, cat in zip((c1, c2), ("Extracurricular_Activity", "Internet_Availability")):
            ct = (pd.crosstab(df[cat], df[TARGET], normalize="index") * 100)
            ct = ct[[c for c in CLASS_ORDER if c in ct.columns]]
            fig = px.bar(ct, barmode="stack", color_discrete_map=COLORS)
            fig.update_layout(title=f"{LABELS[cat]}", yaxis_title="% of students",
                              xaxis_title=LABELS[cat], legend_title_text="")
            with holder:
                show(style(fig, 380))


def page_model():
    m = get_metrics()
    hero("Model performance", "Random Forest results measured on the unseen 20% test data.")
    kpi_row([
        ("Accuracy", f"{m['accuracy'] * 100:.2f}%", "Test data", "🎯"),
        ("Precision (macro)", f"{m['precision_macro'] * 100:.2f}%", "", "🔍"),
        ("Recall (macro)", f"{m['recall_macro'] * 100:.2f}%", "", "📡"),
        ("F1-score (macro)", f"{m['f1_macro'] * 100:.2f}%", "", "⚡"),
    ])
    st.write("")
    kpi_row([
        ("Training accuracy", f"{m['train_accuracy'] * 100:.2f}%", "Seen data", "🏋️"),
        ("5-fold CV accuracy", f"{m['cv_accuracy_mean'] * 100:.2f}%", f"± {m['cv_accuracy_std'] * 100:.2f}", "🧪"),
    ])

    a, b = st.columns(2)
    with a:
        section("Confusion matrix")
        cls = m["classes"]
        fig = px.imshow(np.array(m["confusion_matrix"]), x=cls, y=cls, text_auto=True,
                        color_continuous_scale=[[0, "#0b1a33"], [1, CYAN]],
                        labels=dict(x="Predicted category", y="Actual category"))
        fig.update_layout(coloraxis_showscale=False)
        show(style(fig, 420))
    with b:
        section("Feature importance")
        imp = pd.Series(m["feature_importance"]).sort_values()
        fig = go.Figure(go.Bar(x=imp.values, y=[i.replace("_", " ") for i in imp.index],
                               orientation="h",
                               marker=dict(color=imp.values, colorscale=[[0, "#1d4ed8"], [1, CYAN]])))
        show(style(fig, 420))

    section("Per-class precision, recall and F1")
    rep = pd.DataFrame(m["classification_report"]).T
    rep.index.name = "Class"
    per = rep.loc[[c for c in m["classes"] if c in rep.index], ["precision", "recall", "f1-score"]]
    long = per.reset_index().melt(id_vars="Class", var_name="Metric", value_name="Score")
    fig = px.bar(long, x="Class", y="Score", color="Metric", barmode="group",
                 color_discrete_sequence=[BLUE, CYAN, INDIGO])
    fig.update_layout(yaxis=dict(range=[0, 1]))
    show(style(fig, 360))

    section("Classification report")
    st.dataframe(rep.round(3), use_container_width=True)
    with st.expander("Model settings (hyperparameters)"):
        st.json(m["hyperparameters"])
    with st.expander("What do these metrics mean?"):
        st.markdown(
            "- **Accuracy**: how many students were predicted correctly.\n"
            "- **Precision**: when the model says 'Good', how often it is really Good.\n"
            "- **Recall**: of all students who are really Good, how many the model found.\n"
            "- **F1-score**: balance (harmonic mean) of precision and recall.\n"
            "- **Confusion matrix**: actual vs predicted; the diagonal shows correct predictions.")


def page_dataset():
    raw, clean = get_raw(), get_clean()
    hero("Dataset explorer", "Synthetic dataset stored in dataset/student_performance.csv.")
    kpi_row([
        ("Raw records", f"{len(raw):,}", "", "🗂️"),
        ("After cleaning", f"{len(clean):,}", "", "🧹"),
        ("Input features", f"{len(FEATURES)}", "", "🧬"),
        ("Duplicates found", f"{int(raw.duplicated().sum())}", "", "♻️"),
    ])
    section("Preview (first 10 raw rows)")
    st.dataframe(raw.head(10), use_container_width=True)

    a, b = st.columns(2)
    with a:
        section("Missing values (raw data)")
        miss = raw.isna().sum()
        fig = go.Figure(go.Bar(x=miss.values, y=[LABELS.get(c, c) for c in miss.index],
                               orientation="h", marker=dict(color=BLUE)))
        fig.update_layout(xaxis_title="Missing values")
        show(style(fig, 380))
    with b:
        section("Category distribution")
        counts = clean[TARGET].value_counts().reindex(CLASS_ORDER)
        fig = go.Figure(go.Bar(x=counts.index, y=counts.values,
                               marker=dict(color=[COLORS[c] for c in counts.index]),
                               text=counts.values, textposition="outside"))
        fig.update_layout(yaxis_title="Students")
        show(style(fig, 380))

    section("Statistical summary")
    st.dataframe(clean[NUMERIC_FEATURES].describe().round(2), use_container_width=True)

    section("Column descriptions")
    desc = pd.DataFrame({
        "Column": [LABELS.get(c, c) for c in raw.columns],
        "Data type": [str(t) for t in raw.dtypes],
        "Description": [
            "Unique student identifier (not used for training)", "Average self-study hours per day",
            "Percentage of classes attended", "Marks in the previous examination (0-100)",
            "Average assignment score (0-100)", "Internal assessment marks (0-100)",
            "Average sleep hours per day", "Takes part in extracurricular activities (Yes/No)",
            "Has internet access at home (Yes/No)", "Target label: Good / Average / Poor",
        ],
    })
    st.dataframe(desc, use_container_width=True)


# ----------------------------------------------------------------------
# Sidebar navigation
# ----------------------------------------------------------------------
PAGES = {
    "📊  Overview": page_overview,
    "🔮  Prediction": page_prediction,
    "📈  Analytics": page_analytics,
    "🏆  Model": page_model,
    "🗂️  Dataset": page_dataset,
}

with st.sidebar:
    st.markdown(
        '<div class="brand"><div class="brand-logo">🎓</div><div>'
        '<div class="brand-name">STUDENTIQ</div>'
        '<div class="brand-sub">AI performance analytics</div></div></div>',
        unsafe_allow_html=True)
    choice = st.radio("Navigation", list(PAGES.keys()), label_visibility="collapsed")
    st.markdown(
        '<div class="side-card"><b>Engine</b><br>Random Forest Classifier<br>'
        '<b>Library</b><br>Scikit-learn<br><b>Data</b><br>Synthetic CSV dataset</div>',
        unsafe_allow_html=True)

try:
    PAGES[choice]()
except FileNotFoundError as err:
    st.error(str(err))
    st.info("Run `python generate_dataset.py` and then `python train_model.py`, then refresh this page.")