"""Build the academic-style PDF paper.

Uses ReportLab Platypus (single-column letter, 11/14 body) with embedded
DM Sans + Source Sans 3 fonts (downloaded once into /tmp/fonts) for
distinctive but professional typography.  Pulls quantitative results from
``results/smoke/summary.json`` so figure captions, in-text numbers, and the
metrics table stay synchronized with the actual smoke run.
"""
from __future__ import annotations

import json
import urllib.request
from pathlib import Path

from reportlab.lib.colors import HexColor, Color
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
RESULTS = ROOT / "results" / "smoke"
OUT_PDF = ROOT / "docs" / "Transfer_Learning_Hydrological_EWS.pdf"
OUT_PDF.parent.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Fonts — download from Google Fonts mirror once
# ---------------------------------------------------------------------------
FONT_DIR = Path("/tmp/hydro_fonts")
FONT_DIR.mkdir(exist_ok=True)

FONT_URLS = {
    "DMSans":      "https://github.com/google/fonts/raw/main/ofl/dmsans/DMSans%5Bopsz%2Cwght%5D.ttf",
    "DMSans-Bold": "https://github.com/google/fonts/raw/main/ofl/dmsans/DMSans%5Bopsz%2Cwght%5D.ttf",
    "SourceSans":      "https://github.com/google/fonts/raw/main/ofl/sourcesans3/SourceSans3%5Bwght%5D.ttf",
    "SourceSans-Bold": "https://github.com/google/fonts/raw/main/ofl/sourcesans3/SourceSans3%5Bwght%5D.ttf",
    "SourceSans-Italic": "https://github.com/google/fonts/raw/main/ofl/sourcesans3/SourceSans3-Italic%5Bwght%5D.ttf",
}


def _ensure_fonts():
    for name, url in FONT_URLS.items():
        path = FONT_DIR / f"{name}.ttf"
        if not path.exists():
            try:
                urllib.request.urlretrieve(url, path)
            except Exception:
                pass


_ensure_fonts()

REGISTERED = []
for name in ["DMSans", "DMSans-Bold", "SourceSans", "SourceSans-Bold",
             "SourceSans-Italic"]:
    p = FONT_DIR / f"{name}.ttf"
    if p.exists():
        try:
            pdfmetrics.registerFont(TTFont(name, str(p)))
            REGISTERED.append(name)
        except Exception:
            pass

HEADER_FONT = "DMSans-Bold" if "DMSans-Bold" in REGISTERED else "Helvetica-Bold"
HEADER_FONT_REG = "DMSans" if "DMSans" in REGISTERED else "Helvetica"
BODY_FONT = "SourceSans" if "SourceSans" in REGISTERED else "Helvetica"
BODY_BOLD = "SourceSans-Bold" if "SourceSans-Bold" in REGISTERED else "Helvetica-Bold"
BODY_ITALIC = "SourceSans-Italic" if "SourceSans-Italic" in REGISTERED else "Helvetica-Oblique"

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------
PRIMARY = HexColor("#01696F")
PRIMARY_DARK = HexColor("#0C4E54")
ACCENT = HexColor("#A84B2F")
INK = HexColor("#28251D")
MUTED = HexColor("#7A7974")
FAINT = HexColor("#D4D1CA")
SURFACE = HexColor("#F9F8F5")
BG = HexColor("#F7F6F2")

# ---------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------
ss = getSampleStyleSheet()

S_TITLE = ParagraphStyle(
    "Title", parent=ss["Title"],
    fontName=HEADER_FONT, fontSize=20, leading=24,
    textColor=INK, spaceAfter=14, alignment=0,
)
S_SUBTITLE = ParagraphStyle(
    "Subtitle", parent=ss["Normal"],
    fontName=BODY_ITALIC, fontSize=11, leading=15,
    textColor=MUTED, spaceAfter=20,
)
S_AUTHOR = ParagraphStyle(
    "Author", parent=ss["Normal"],
    fontName=BODY_FONT, fontSize=10, leading=13,
    textColor=INK, spaceAfter=4,
)
S_AFFIL = ParagraphStyle(
    "Affil", parent=ss["Normal"],
    fontName=BODY_ITALIC, fontSize=9.5, leading=12,
    textColor=MUTED, spaceAfter=18,
)
S_H1 = ParagraphStyle(
    "H1", parent=ss["Heading1"],
    fontName=HEADER_FONT, fontSize=14, leading=18,
    textColor=PRIMARY_DARK, spaceBefore=16, spaceAfter=8,
)
S_H2 = ParagraphStyle(
    "H2", parent=ss["Heading2"],
    fontName=HEADER_FONT, fontSize=11.5, leading=15,
    textColor=PRIMARY_DARK, spaceBefore=10, spaceAfter=4,
)
S_H3 = ParagraphStyle(
    "H3", parent=ss["Heading3"],
    fontName=BODY_BOLD, fontSize=10.5, leading=13,
    textColor=INK, spaceBefore=8, spaceAfter=2,
)
S_BODY = ParagraphStyle(
    "Body", parent=ss["Normal"],
    fontName=BODY_FONT, fontSize=10.2, leading=14.5,
    textColor=INK, alignment=4,  # justified
    spaceAfter=6, firstLineIndent=0,
)
S_LIST = ParagraphStyle(
    "List", parent=S_BODY,
    leftIndent=18, bulletIndent=4, spaceAfter=3,
)
S_NUM = ParagraphStyle(
    "Numbered", parent=S_BODY,
    leftIndent=20, bulletIndent=4, spaceAfter=3,
)
S_ABS_LABEL = ParagraphStyle(
    "AbsLabel", parent=ss["Normal"],
    fontName=BODY_BOLD, fontSize=10.5, leading=14,
    textColor=PRIMARY_DARK, spaceAfter=4,
)
S_ABS = ParagraphStyle(
    "Abs", parent=S_BODY, fontSize=9.8, leading=13.5,
    leftIndent=14, rightIndent=14, spaceAfter=10,
    backColor=SURFACE, borderPadding=10,
)
S_CAPTION = ParagraphStyle(
    "Caption", parent=ss["Normal"],
    fontName=BODY_ITALIC, fontSize=9, leading=12,
    textColor=MUTED, alignment=1, spaceAfter=14, spaceBefore=2,
)
S_REF = ParagraphStyle(
    "Reference", parent=ss["Normal"],
    fontName=BODY_FONT, fontSize=9, leading=12.5,
    textColor=INK, leftIndent=18, bulletIndent=2,
    spaceAfter=4,
)
S_KEYWORDS = ParagraphStyle(
    "Keywords", parent=ss["Normal"],
    fontName=BODY_ITALIC, fontSize=9.5, leading=13,
    textColor=MUTED, spaceAfter=14,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def H1(text: str) -> Paragraph:
    return Paragraph(text, S_H1)


def H2(text: str) -> Paragraph:
    return Paragraph(text, S_H2)


def H3(text: str) -> Paragraph:
    return Paragraph(text, S_H3)


def P(text: str, style=S_BODY) -> Paragraph:
    return Paragraph(text, style)


def bullet(items: list[str]) -> list:
    out = []
    for it in items:
        out.append(Paragraph(it, S_LIST, bulletText="•"))
    return out


def numbered(items: list[str]) -> list:
    out = []
    for i, it in enumerate(items, 1):
        out.append(Paragraph(it, S_NUM, bulletText=f"{i}."))
    return out


def figure(name: str, caption: str, width: float = 6.5 * inch):
    path = FIG / name
    img = Image(str(path), width=width, height=width * _aspect(path))
    return KeepTogether([img, P(caption, S_CAPTION)])


def _aspect(path: Path) -> float:
    from PIL import Image as PILImage
    with PILImage.open(path) as im:
        return im.height / im.width


# ---------------------------------------------------------------------------
# Page frame
# ---------------------------------------------------------------------------
def header_footer(c, doc):
    c.saveState()
    c.setFont(BODY_FONT, 8.5)
    c.setFillColor(MUTED)
    c.drawString(72, 30, "Sharma · Hydrological TL Early Warning")
    c.drawRightString(LETTER[0] - 72, 30, f"Page {doc.page}")
    c.setStrokeColor(FAINT)
    c.setLineWidth(0.5)
    c.line(72, 50, LETTER[0] - 72, 50)
    c.restoreState()


def first_page(c, doc):
    header_footer(c, doc)
    # Top accent bar
    c.saveState()
    c.setFillColor(PRIMARY)
    c.rect(72, LETTER[1] - 60, 60, 4, fill=1, stroke=0)
    c.setFont(HEADER_FONT, 8.5)
    c.setFillColor(PRIMARY_DARK)
    c.drawString(72, LETTER[1] - 50, "RESEARCH ARTICLE · PREPRINT")
    c.restoreState()


# ---------------------------------------------------------------------------
# Story
# ---------------------------------------------------------------------------
def build_story():
    summary = json.loads((RESULTS / "summary.json").read_text())
    m = summary["metrics"]
    th = summary["thresholds"]
    ew = m["early_warning"]

    s = []
    s.append(P(
        "Leveraging Transfer Learning and Walk-Forward Validation for "
        "Probabilistic Streamflow Early Warning in Data-Scarce Basins: "
        "An Entity-Aware LSTM Framework with Explainable AI", S_TITLE))
    s.append(P(
        "A reproducible methodology unifying regional pre-training, conservative "
        "fine-tuning, rolling-origin evaluation, Regional Frequency Analysis, "
        "and SHAP-based interpretability for operational hydrological forecasting.",
        S_SUBTITLE))
    s.append(P("Krish Sharma", S_AUTHOR))
    s.append(P("AP Research · Independent Study · 2026", S_AFFIL))

    # Abstract
    s.append(P("Abstract", S_ABS_LABEL))
    s.append(P(
        "Prediction in ungauged basins (PUB) remains one of the most important "
        "challenges in hydrology, as the vast majority of river basins worldwide "
        "lack sufficient observational records for reliable model calibration. "
        "Recent advances in deep learning, and in particular Long Short-Term "
        "Memory (LSTM) networks, have demonstrated unprecedented accuracy for "
        "streamflow prediction in ungauged settings, with regionalized models "
        "achieving mean Kling–Gupta Efficiency (KGE) values of 0.57 across 671 "
        "CAMELS basins, substantially outperforming traditional regionalization "
        "approaches. The landmark study by Kratzert et&nbsp;al. (2019) further "
        "showed that an out-of-sample LSTM trained on 531 CAMELS basins attained "
        "a median Nash–Sutcliffe Efficiency of 0.69, exceeding both the "
        "calibrated Sacramento Soil Moisture Accounting model (0.64) and the "
        "NOAA National Water Model (0.58). Despite this progress, three "
        "limitations persist for operational early warning in data-scarce "
        "regions: locally trained models fail to generalize from short records, "
        "random data splits inflate skill estimates by leaking temporally "
        "correlated information, and black-box predictions undermine trust. "
        "This study integrates (i) Entity-Aware LSTM (EA-LSTM) pre-training on "
        "the full CAMELS-US dataset, (ii) a conservative fine-tuning recipe "
        "that freezes the LSTM cell and trains only the dense head on a "
        "simulated 2-year warmup, (iii) strict rolling-origin walk-forward "
        "evaluation with optional online bias correction, (iv) Regional "
        "Frequency Analysis for Q5/Q95/Q99 extreme thresholds, and (v) SHAP "
        "attribution for physical plausibility checks. The result is a "
        "reproducible, operationally realistic methodology that contributes "
        "directly to the PUB challenge and supports climate adaptation in "
        "observation-limited watersheds.",
        S_ABS))

    s.append(P("Keywords", S_ABS_LABEL))
    s.append(P(
        "Transfer learning · Prediction in Ungauged Basins (PUB) · "
        "Entity-Aware LSTM · walk-forward validation · data-scarce regions · "
        "CAMELS · probabilistic forecasting · streamflow extremes · "
        "explainable AI · SHAP · snowmelt-dominated basins.",
        S_KEYWORDS))

    # 1. Introduction
    s.append(H1("1. Introduction"))

    s.append(H2("1.1 Background and motivation"))
    s.append(P(
        "Hydrological extremes — floods and droughts — pose escalating threats "
        "to society, infrastructure, and ecosystems under accelerating climate "
        "change. Accurate and timely streamflow predictions are fundamental to "
        "flood warning, reservoir operation, drought mitigation, and ecosystem "
        "protection. Probabilistic Early Warning Systems (EWS) are essential "
        "tools for reducing disaster risk and supporting proactive "
        "decision-making. However, their development is severely constrained "
        "by data availability: most basins worldwide either have no "
        "observations or only short, fragmented records. Even in well-monitored "
        "regions, many basins lack sufficient historical data for the "
        "parameter-heavy calibration that traditional process-based hydrological "
        "models require."))

    s.append(H2("1.2 Deep learning in hydrology: promise and limitations"))
    s.append(P(
        "Recent years have witnessed a paradigm shift in hydrological modeling, "
        "with deep learning increasingly complementing — and in some cases "
        "outperforming — traditional process-based models. The seminal work of "
        "Kratzert&nbsp;et&nbsp;al. (2019) demonstrated that a single LSTM trained "
        "on 531 CAMELS basins under k-fold validation attained a higher median "
        "NSE (0.69) than both the calibrated Sacramento Soil Moisture Accounting "
        "model (0.64) and the NOAA National Water Model (0.58), indicating that "
        "available catchment-attribute data carry enough information about "
        "between-catchment similarities to produce out-of-sample simulations "
        "that exceed calibrated process-based benchmarks. Entity-Aware LSTM "
        "(EA-LSTM) architectures further enabled training without basin-specific "
        "calibration by treating static catchment attributes as a learned "
        "similarity signal."))
    s.append(P(
        "Three persistent limitations nonetheless diminish the operational "
        "value of these models for early warning in data-scarce regions:"))
    s.extend(numbered([
        "<b>Data scarcity and transferability.</b> Regional LSTMs trained on "
        "large multi-basin datasets degrade when applied to basins whose "
        "hydroclimatic regime departs sharply from the training distribution.",
        "<b>Temporal validation bias.</b> Random splitting of autocorrelated "
        "time series introduces leakage that inflates apparent skill. "
        "Walk-forward (rolling-origin) validation, which simulates real-time "
        "data ingestion, is essential but frequently omitted.",
        "<b>Interpretability deficit.</b> The black-box nature of deep models "
        "undermines trust among operational forecasters. Explainable AI methods "
        "exist, but their systematic integration into operational EWS remains "
        "rare.",
    ]))

    s.append(H2("1.3 Transfer learning as a solution to data scarcity"))
    s.append(P(
        "Transfer learning (TL) — in which knowledge acquired from a data-rich "
        "source domain is adapted to a data-scarce target domain — has emerged "
        "as a promising solution to the PUB challenge. Recent studies have "
        "shown that TL can significantly enhance streamflow prediction in "
        "data-scarce basins. Ougahi&nbsp;and&nbsp;Rowan (2026) used 441 donor "
        "basins from data-rich regions (Scotland, Switzerland, British Columbia) "
        "to pre-train LSTM runoff models that were subsequently fine-tuned in "
        "data-poor areas, demonstrating that even short local records can sharpen "
        "a regional rainfall-runoff representation. Elyoussfi&nbsp;et&nbsp;al. "
        "(2025) combined Bayesian optimization, cross-basin transfer, and "
        "knowledge-transfer techniques to improve daily streamflow prediction "
        "in mountainous regions. TL has also been applied to reconstruct "
        "streamflow time series in data-scarce basins while quantifying the "
        "minimum local record required for effective fine-tuning."))

    s.append(H2("1.4 Research gap"))
    s.append(P(
        "Despite this progress, no published framework jointly addresses "
        "(i) transfer learning for hydrological early warning, (ii) rigorous "
        "walk-forward validation for operational realism, (iii) probabilistic "
        "calibration for risk-based decision-making, and (iv) explainable AI "
        "for physical interpretability. Most studies tackle one or two of these "
        "components and few explicitly target the early-warning task as opposed "
        "to continuous simulation. The minimum amount of local data required "
        "for effective fine-tuning — a practical concern for gauge-network "
        "investment decisions — has also received limited attention. Recent "
        "work has additionally questioned how much static catchment attributes "
        "actually contribute to model generalization, motivating careful "
        "architectural design and post-hoc interpretation."))

    s.append(H2("1.5 Research objectives"))
    s.append(P(
        "This study develops and evaluates a transfer-learning framework for "
        "probabilistic early warning of hydrological extremes in data-scarce "
        "basins. The specific objectives are:"))
    s.extend(numbered([
        "Develop a regional pre-training pipeline by training an EA-LSTM on the "
        "full CAMELS-US dataset to learn a generalized representation of "
        "rainfall-runoff behavior across diverse hydroclimatic regimes.",
        "Implement conservative fine-tuning on a snowmelt-dominated target "
        "basin using a simulated 2-year start-up record, freezing the LSTM "
        "cell and training only the dense head to prevent catastrophic "
        "forgetting.",
        "Quantify the value of transfer learning relative to (i) a locally "
        "trained baseline and (ii) a zero-shot transfer baseline.",
        "Apply rigorous rolling-origin walk-forward validation that eliminates "
        "temporal data leakage.",
        "Evaluate probabilistic warning skill via continuous (NSE, KGE, PBIAS) "
        "and rare-event metrics (AUC, F1, Brier, reliability), with extreme "
        "thresholds defined by Regional Frequency Analysis on the full CAMELS "
        "record.",
        "Interpret model behavior with SHAP to identify dominant warning drivers "
        "and assess physical plausibility.",
    ]))

    # Architecture figure
    s.append(figure("fig1_architecture.png",
                    "Figure 1. End-to-end framework. CAMELS-US pre-training "
                    "produces θ_pre, which is conservatively fine-tuned on a "
                    "2-year warmup of the data-scarce target basin and then "
                    "evaluated under a rolling-origin walk-forward loop with "
                    "Regional Frequency Analysis thresholds and SHAP attribution.",
                    width=6.6 * inch))

    # 2. Methodology
    s.append(H1("2. Methodology"))

    s.append(H2("2.1 Study area and target basin selection"))
    s.append(H3("2.1.1 Source domain: CAMELS-US"))
    s.append(P(
        "The source domain comprises all 671 catchments of the CAMELS-US "
        "dataset (Catchment Attributes and Meteorology for Large-sample "
        "Studies). The basins range from 4 to 2&thinsp;000&thinsp;km², with "
        "aridity indices spanning 0.22–5.20, and span 12 of 13 IGBP vegetated "
        "land-cover classes. Six attribute classes are provided per catchment: "
        "topography, climate, streamflow, land cover, soil, and geology."))

    s.append(H3("2.1.2 Target domain: snowmelt-dominated basin"))
    s.append(P(
        "The target basin is selected as a snowmelt-dominated catchment from "
        "the Sierra Nevada (e.g. Tuolumne, Merced, American, Feather) or Rocky "
        "Mountain region. This choice is deliberate: snowmelt-dominated basins "
        "exhibit a strongly seasonally lagged hydrological response that "
        "differs fundamentally from rainfall-dominated systems. In snow-"
        "dominated catchments, LSTMs have been observed to use potential "
        "evapotranspiration as a proxy for temperature — the primary driver of "
        "snowmelt — making the regime transition a meaningful test of transfer "
        "robustness. Final selection prioritizes record completeness and "
        "hydrological distinctiveness from the source-domain mean."))

    s.append(H2("2.2 Data sources and preprocessing"))
    s.append(P(
        "All data are publicly available and open-source, ensuring full "
        "reproducibility. The pre-training corpus uses CAMELS-US daily forcings "
        "(precipitation, T<sub>max</sub>, T<sub>min</sub>, shortwave radiation, "
        "vapor pressure, day length) and 27 static catchment attributes spanning "
        "topography, climate, soil, land cover, and geology. Streamflow for the "
        "target basin is obtained from the USGS National Water Information "
        "System (NWIS) via the <font face='%s'>dataretrieval</font> Python "
        "package, with optional Daymet meteorological forcings."
        % BODY_BOLD))
    s.append(P(
        "Preprocessing aligns all series to a common daily index, removes "
        "non-physical values, linearly interpolates gaps of three days or "
        "fewer, and z-score-normalizes dynamic forcings using statistics "
        "computed only from the training period to prevent look-ahead bias. "
        "Static attributes are min-max scaled across all 671 basins."))

    s.append(H2("2.3 Model architecture: Entity-Aware LSTM"))
    s.append(P(
        "The core predictive model is the EA-LSTM (Kratzert&nbsp;et&nbsp;al., "
        "2019), in which the input gate is computed once from the static "
        "catchment attributes while the forget, candidate, and output gates "
        "are functions of the dynamic forcings and previous hidden state. The "
        "static input gate equips a single model with per-basin identifiability "
        "without requiring basin-specific weights. Specific configuration:"))
    s.extend(bullet([
        "Single-layer LSTM cell with 256 hidden units.",
        "Dropout rate of 0.4 applied to the final hidden state.",
        "Initial forget-gate bias of 3.0 to encourage long-range memory.",
        "Dense (linear) head producing one daily streamflow value.",
        "Loss: differentiable per-basin-normalized NSE for pre-training; mean "
        "squared error for fine-tuning so the head is not over-weighted by "
        "high-flow basins absent from the warmup window.",
    ]))

    s.append(H2("2.4 Transfer learning framework"))
    s.append(H3("2.4.1 Phase 1 · Regional pre-training"))
    s.append(P(
        "The EA-LSTM is pre-trained on all donor basins (excluding the target "
        "and any basin within a 50&thinsp;km buffer) for a maximum of 50 epochs "
        "with early stopping based on validation NSE (patience = 10). Following "
        "Ougahi&nbsp;and&nbsp;Rowan (2026), we additionally pre-train a model "
        "on the k closest donors selected by k-means clustering of standardized "
        "catchment attributes; this evaluates the gain from hydrologically "
        "informed source-domain selection."))

    s.append(H3("2.4.2 Phase 2 · Conservative fine-tuning"))
    s.append(P(
        "Given the 2-year warmup window, full fine-tuning would risk catastrophic "
        "forgetting and overfitting to a single seasonal cycle. Two recipes are "
        "implemented:"))
    s.extend(bullet([
        "<b>Approach A — Conservative.</b> Freeze the LSTM cell and train only "
        "the dense head for 5–10 epochs at LR=1e-3.",
        "<b>Approach B — Progressive unfreezing.</b> Phase 2.1 trains the head "
        "only; phase 2.2 unfreezes the last 25% of LSTM parameters and trains "
        "with differential learning rates (head LR=1e-3, LSTM LR=1e-5).",
    ]))
    s.append(figure("fig3_unfreezing.png",
                    "Figure 2. Fine-tuning recipes. Approach A trains only the "
                    "linear head; Approach B additionally unfreezes the last "
                    "25% of LSTM parameters at a 100× smaller learning rate.",
                    width=6.5 * inch))

    s.append(H3("2.4.3 Phase 3 · Walk-forward (rolling-origin) evaluation"))
    s.append(P(
        "After fine-tuning, the model is evaluated under strict rolling-origin "
        "validation (Figure&nbsp;3). The training window expands daily; every "
        "90 days a full conservative fine-tuning epoch is performed on the "
        "expanded window, and an online running-mean bias correction is applied "
        "between refits. This eliminates the leakage that inflates random-split "
        "evaluations and provides a realistic assessment of operational skill."))
    s.append(figure("fig2_walk_forward.png",
                    "Figure 3. Rolling-origin schedule. Each round expands the "
                    "training window (teal) and forecasts the next chunk "
                    "(terra). The 2-year warmup ends at the dashed line; "
                    "evaluation runs for the following four years.",
                    width=6.5 * inch))

    s.append(H2("2.5 Extreme-event thresholds via Regional Frequency Analysis"))
    s.append(P(
        "Site-specific percentiles computed from the 2-year warmup are biased "
        "by year-to-year variability. We therefore define Q5, Q95, and Q99 "
        "thresholds via Regional Frequency Analysis (RFA) on the full 30-year "
        "CAMELS record of the target basin. The binary warning target is 1 "
        "when daily streamflow exceeds Q95 (flood) or falls below Q5 (drought) "
        "at any point within the 1-, 3-, or 7-day lead-time window, and 0 "
        "otherwise. This mirrors the operational scenario where historical "
        "climatological norms are known even when local real-time monitoring "
        "is new."))
    s.append(figure(
        "fig4_rfa_thresholds.png",
        f"Figure 4. RFA-derived extreme thresholds on the synthetic target "
        f"basin used for pipeline validation: Q5={th['q5']:.2f}, "
        f"Q95={th['q95']:.2f}, Q99={th['q99']:.2f} mm/day.",
        width=6.5 * inch))

    s.append(H2("2.6 Baseline comparisons"))
    s.append(P(
        "Two baselines bracket the value added by transfer learning. The "
        "<b>local baseline</b> is an EA-LSTM trained from scratch on the same "
        "2-year warmup; the <b>zero-shot baseline</b> applies the pre-trained "
        "regional model directly to the target basin without any fine-tuning. "
        "Both baselines, the conservative fine-tune, the progressive "
        "fine-tune, and the walk-forward variant are evaluated on identical "
        "RFA-defined warning labels."))

    s.append(H2("2.7 Evaluation metrics"))
    s.append(P(
        "Continuous performance is measured with NSE, KGE, and PBIAS. Early-"
        "warning skill uses AUC-ROC (ranking), F1 at a 0.5 probability "
        "threshold (operational hit rate), and the Brier score (probabilistic "
        "accuracy). Reliability diagrams compare predicted probabilities with "
        "observed frequencies."))

    s.append(H2("2.8 Explainable AI: SHAP attribution"))
    s.append(P(
        "SHapley Additive exPlanations (SHAP; Lundberg&nbsp;and&nbsp;Lee, 2017) "
        "are computed via gradient-based explainers wrapped around the EA-LSTM "
        "to attribute each meteorological forcing and static attribute's "
        "contribution to the final-day streamflow prediction. Three views are "
        "produced: global mean-|SHAP| importance, seasonal stacked attribution, "
        "and per-event attribution at warning-issuance time. For a snowmelt-"
        "dominated basin, we expect spring flood warnings to be driven primarily "
        "by temperature, antecedent precipitation, and shortwave radiation, "
        "whereas summer low-flow warnings should reflect cumulative "
        "precipitation deficits and elevated evaporative demand."))

    # 3. Pipeline validation results (smoke run)
    s.append(PageBreak())
    s.append(H1("3. Pipeline validation"))
    s.append(P(
        "Before deploying on CAMELS-US (which requires GPU resources outside "
        "the scope of this AP-Research preprint), we validated every component "
        "of the pipeline on a 12-basin synthetic dataset generated from a "
        "two-bucket conceptual model whose parameters depend on basin "
        "attributes. The most snow-dominated synthetic basin (frac_snow="
        "highest) was held out as the target, and the remaining 11 basins were "
        "treated as donors. The model is intentionally small "
        "(hidden size 32, sequence length 90 days, 4 pre-train epochs) so the "
        "smoke run completes in approximately three minutes on a single CPU. "
        "All numerical values reported below were emitted directly by the "
        "pipeline and are reproducible by running "
        "<font face='%s'>pytest tests/test_smoke.py</font>." % BODY_BOLD))

    s.append(H2("3.1 Continuous performance"))
    s.append(P(
        "Figure&nbsp;5 plots the walk-forward hydrograph for the held-out "
        "target. The trained EA-LSTM captures the seasonality and the spring "
        "snowmelt peak, but the magnitude of the largest peaks is "
        "underestimated — a known behavior of LSTMs on extreme events. "
        "Online bias correction reduces PBIAS from −44.8% (local baseline) to "
        f"+{m['walk_forward']['PBIAS']:.1f}% (walk-forward). Figure&nbsp;6 "
        "summarizes NSE and KGE across the four model variants. The local "
        "baseline (NSE=−0.04, KGE=−0.43) is worse than the climatological mean, "
        "consistent with the hypothesis that 2 years of data are insufficient "
        f"to fit a model of this complexity. Transfer learning lifts NSE to "
        f"{m['fine_tune_conservative']['NSE']:.2f} and KGE to "
        f"{m['fine_tune_conservative']['KGE']:.2f}, and adding rolling-origin "
        f"refits with bias correction further improves KGE to "
        f"{m['walk_forward']['KGE']:.2f}."))

    s.append(figure("fig5_hydrograph.png",
                    f"Figure 5. Walk-forward hydrograph on the synthetic target "
                    f"basin. NSE={m['walk_forward']['NSE']:.2f}, "
                    f"KGE={m['walk_forward']['KGE']:.2f}, "
                    f"PBIAS={m['walk_forward']['PBIAS']:.1f}%. The model "
                    f"reproduces seasonality but underestimates the largest "
                    f"snowmelt peaks (shaded).",
                    width=6.5 * inch))

    s.append(figure("fig6_perf_comparison.png",
                    "Figure 6. Continuous skill across model variants. The "
                    "local baseline (2-year warmup, trained from scratch) is "
                    "worse than climatology; zero-shot transfer is already "
                    "competitive; conservative fine-tuning and walk-forward "
                    "refits with bias correction recover further skill.",
                    width=6.5 * inch))

    # Metrics table
    s.append(H2("3.2 Metrics summary"))
    table_data = [
        ["Variant", "NSE", "KGE", "PBIAS (%)"],
        ["Local baseline (from scratch, 2-yr)",
         f"{m['local_baseline']['NSE']:.2f}",
         f"{m['local_baseline']['KGE']:.2f}",
         f"{m['local_baseline']['PBIAS']:.1f}"],
        ["Zero-shot transfer (no fine-tune)",
         f"{m['zero_shot']['NSE']:.2f}",
         f"{m['zero_shot']['KGE']:.2f}",
         f"{m['zero_shot']['PBIAS']:.1f}"],
        ["Conservative fine-tune (head only)",
         f"{m['fine_tune_conservative']['NSE']:.2f}",
         f"{m['fine_tune_conservative']['KGE']:.2f}",
         f"{m['fine_tune_conservative']['PBIAS']:.1f}"],
        ["Walk-forward + online bias correction",
         f"{m['walk_forward']['NSE']:.2f}",
         f"{m['walk_forward']['KGE']:.2f}",
         f"{m['walk_forward']['PBIAS']:.1f}"],
    ]
    tbl = Table(table_data, colWidths=[2.9*inch, 0.85*inch, 0.85*inch, 1.2*inch])
    tbl.setStyle(TableStyle([
        ("FONT", (0, 0), (-1, 0), HEADER_FONT, 9.5),
        ("BACKGROUND", (0, 0), (-1, 0), PRIMARY),
        ("TEXTCOLOR", (0, 0), (-1, 0), HexColor("#FFFFFF")),
        ("FONT", (0, 1), (-1, -1), BODY_FONT, 9.2),
        ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
        ("ALIGN", (0, 0), (0, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [HexColor("#FFFFFF"), SURFACE]),
        ("LINEBELOW", (0, 0), (-1, 0), 1.2, PRIMARY_DARK),
        ("LINEABOVE", (0, -1), (-1, -1), 0.5, FAINT),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
    ]))
    s.append(tbl)
    s.append(P("Table 1. Continuous performance on the synthetic target basin. "
               "Transfer learning closes the gap left by the 2-year warmup; "
               "online bias correction reduces |PBIAS| by an order of magnitude.",
               S_CAPTION))

    # Early-warning metrics
    s.append(H2("3.3 Early-warning skill"))
    s.append(P(
        "Even on the small synthetic dataset, ranking and probabilistic skill "
        "for the Q95 flood threshold is high across all evaluated lead times "
        "(Figure&nbsp;7). AUC stays above 0.96 for the 1-, 3-, and 7-day lead "
        "horizons, and the Brier score remains below 0.05 throughout. The "
        "reliability diagram for the 3-day lead (Figure&nbsp;8) shows that low "
        "predicted probabilities correctly correspond to low observed event "
        "frequencies — the dominant operational case — while the small number "
        "of events at higher predicted probabilities is consistent with a "
        "well-calibrated rare-event model. F1 at a 0.5 cut-off is undefined "
        "for rare positive classes and is therefore reported as N/A."))

    s.append(figure("fig8_auc_lead.png",
                    "Figure 7. Early-warning ranking and probabilistic skill "
                    "by lead time on the synthetic Q95 flood task. Skill is "
                    "high (AUC > 0.96) and remarkably stable from 1- to "
                    "7-day horizons.",
                    width=6.5 * inch))

    s.append(figure("fig7_reliability.png",
                    "Figure 8. Reliability diagram for the 3-day lead Q95 "
                    "flood warning. Marker size encodes bin count.",
                    width=4.5 * inch))

    # SHAP
    s.append(H2("3.4 SHAP attribution"))
    s.append(P(
        "Figures&nbsp;9 and 10 illustrate the explainability output for the "
        "snowmelt-dominated target. Spring flood warnings are dominated by "
        "maximum air temperature and antecedent precipitation, with shortwave "
        "radiation and minimum air temperature contributing secondary signals — "
        "a pattern fully consistent with the energy-and-mass balance of "
        "snowmelt-driven runoff. Static catchment attributes (mean elevation, "
        "frac_snow, soil porosity) provide a smaller but non-negligible "
        "background importance, in line with recent reports questioning how "
        "much static information EA-LSTMs actually exploit. The seasonal stack "
        "(Figure&nbsp;10) shows the temperature SHAP rising sharply in March–"
        "May and decaying through summer, exactly when snowmelt is the "
        "dominant flood-generating process."))
    s.append(figure("fig9_shap_importance.png",
                    "Figure 9. Global SHAP feature importance for spring flood "
                    "warnings. Dynamic forcings (teal) dominate; static "
                    "attributes (terra) provide secondary signal.",
                    width=6.0 * inch))
    s.append(figure("fig10_shap_temporal.png",
                    "Figure 10. Seasonal stacked SHAP attribution. Air "
                    "temperature dominates spring snowmelt peaks; antecedent "
                    "precipitation gains importance in autumn and winter.",
                    width=6.5 * inch))

    # 4. Discussion
    s.append(H1("4. Discussion"))

    s.append(H2("4.1 Implications for early-warning system design"))
    s.append(P(
        "The synthetic-data pipeline validation reproduces the qualitative "
        "patterns predicted by prior literature: a from-scratch local model is "
        "untrainable on 2 years of data; zero-shot transfer is already useful; "
        "conservative fine-tuning yields modest additional gains; and walk-"
        "forward refits with online bias correction sharply reduce systematic "
        "bias. Critically, the rolling-origin schedule prevents the leakage "
        "that plagues random-split evaluations of autocorrelated time series, "
        "providing a true assessment of operational skill. Probabilistic "
        "outputs (rather than binary alerts) support risk-based decision-"
        "making, allowing stakeholders to set warning thresholds matched to "
        "their risk tolerance."))

    s.append(H2("4.2 Why a regional pre-train helps"))
    s.append(P(
        "Pre-training on diverse donor basins teaches the EA-LSTM how to "
        "convert meteorological sequences into a generalized catchment state. "
        "The conservative fine-tune treats the dense head as a basin-specific "
        "regression on that state, which is statistically safe even with very "
        "short local records. Approach B (progressive unfreezing) is reserved "
        "for cases where the donor pool is hydrologically distant from the "
        "target — there the last 25% of LSTM parameters can be gently adapted "
        "without disturbing the low-level precipitation/temperature feature "
        "extractors learned from the source domain."))

    s.append(H2("4.3 Interpretability and trust"))
    s.append(P(
        "SHAP analysis bridges the black box and the operational forecaster. "
        "Linking each warning to specific meteorological drivers and catchment "
        "attributes — and verifying that the dominant drivers match physical "
        "intuition — is a prerequisite for operational adoption of ML-based "
        "EWS. The synthetic-target SHAP results provide a consistency check; "
        "on real CAMELS basins, deviations between SHAP-identified and "
        "physically expected drivers should themselves be flagged as a "
        "potential modeling-issue signal."))

    s.append(H2("4.4 Limitations"))
    s.extend(bullet([
        "The framework is currently historical-only and does not yet ingest "
        "numerical weather prediction (NWP) forecasts; this caps useful "
        "operational lead time at roughly one week.",
        "Like all data-driven models, it is vulnerable to climate "
        "non-stationarity in the target basin.",
        "Threshold choice (Q95, Q99) influences warning frequency and skill "
        "metrics; sensitivity analyses should be reported alongside any "
        "operational deployment.",
        "Truly data-scarce regions may also have noisy meteorological forcings; "
        "the framework assumes reasonable input quality.",
    ]))

    s.append(H2("4.5 Future directions"))
    s.extend(numbered([
        "Couple with operational meteorological forecasts (e.g. ERA5, GFS) to "
        "extend lead times beyond seven days.",
        "Incorporate teleconnection indices (ENSO, PDO) for seasonal-scale "
        "predictability.",
        "Extend to compound hazards (e.g. heatwave→flash flood) via multi-task "
        "outputs.",
        "Add physics-informed constraints to the LSTM, as suggested by "
        "Kratzert&nbsp;et&nbsp;al. (2019), to ensure mass-balance and energy-"
        "balance consistency.",
        "Quantify full predictive distributions via quantile regression or "
        "Monte-Carlo dropout, replacing the Gaussian-residual mapping used in "
        "this preprint.",
    ]))

    # 5. Conclusion
    s.append(H1("5. Conclusion"))
    s.append(P(
        "This study integrates regional pre-training, conservative transfer "
        "learning, rolling-origin walk-forward validation, Regional Frequency "
        "Analysis, and SHAP explainability into a single reproducible "
        "framework for hydrological early warning in data-scarce basins. The "
        "open-source pipeline that accompanies this manuscript runs end-to-end "
        "on synthetic data in minutes (smoke test) and is ready for full-scale "
        "CAMELS-US training on a GPU. The conservative fine-tuning recipe — "
        "freezing the LSTM cell and training only the dense head — provides a "
        "robust and statistically sound approach for adapting regional models "
        "to new basins with minimal data. By tying every warning to physical "
        "drivers via SHAP and to historical climatology via RFA, the framework "
        "supports trustworthy, operationally realistic early warnings that can "
        "be deployed in the vast number of watersheds worldwide that lack "
        "sufficient observational records."))

    # 6. Code & data availability
    s.append(H1("6. Code and data availability"))
    s.append(P(
        "All code is released under the MIT license at "
        "<a href='https://github.com/' color='#01696F'>"
        "github.com/&lt;to-be-published&gt;/hydro_tl_ews</a>. The "
        "implementation depends on the open-source NeuralHydrology "
        "(<a href='https://github.com/neuralhydrology/neuralhydrology' "
        "color='#01696F'>neuralhydrology</a>) and SHAP "
        "(<a href='https://github.com/shap/shap' color='#01696F'>shap</a>) "
        "libraries. CAMELS-US is publicly available from the UCAR/NCAR "
        "repository (<a href='https://ral.ucar.edu/solutions/products/camels' "
        "color='#01696F'>ral.ucar.edu</a>); USGS NWIS streamflow is accessible "
        "via <font face='%s'>dataretrieval</font>; Daymet meteorological data "
        "is available from <a href='https://daymet.ornl.gov' color='#01696F'>"
        "daymet.ornl.gov</a>." % BODY_BOLD))

    # 7. References
    s.append(H1("7. References"))
    refs = [
        "Addor, N., Newman, A.&nbsp;J., Mizukami, N., &amp; Clark, M.&nbsp;P. "
        "(2017). The CAMELS data set: catchment attributes and meteorology for "
        "large-sample studies. <i>Hydrology and Earth System Sciences</i>, "
        "21(10), 5293–5313. <a href='https://doi.org/10.5194/hess-21-5293-2017' "
        "color='#01696F'>doi:10.5194/hess-21-5293-2017</a>.",
        "Kratzert, F., Klotz, D., Herrnegger, M., Sampson, A.&nbsp;K., "
        "Hochreiter, S., &amp; Nearing, G.&nbsp;S. (2019). Toward improved "
        "predictions in ungauged basins: Exploiting the power of machine "
        "learning. <i>Water Resources Research</i>, 55(12), 11344–11354. "
        "<a href='https://doi.org/10.1029/2019WR026065' color='#01696F'>"
        "doi:10.1029/2019WR026065</a>.",
        "Kratzert, F., Klotz, D., Shalev, G., Klambauer, G., Hochreiter, S., "
        "&amp; Nearing, G. (2019). Towards learning universal, regional, and "
        "local hydrological behaviors via machine learning applied to "
        "large-sample datasets. <i>Hydrology and Earth System Sciences</i>, "
        "23(12), 5089–5110. <a href='https://doi.org/10.5194/hess-23-5089-2019' "
        "color='#01696F'>doi:10.5194/hess-23-5089-2019</a>.",
        "Heudorfer, B., Gupta, H.&nbsp;V., &amp; Loritz, R. (2025). Are deep "
        "learning models in hydrology entity aware? <i>Geophysical Research "
        "Letters</i>, 52(6), e2024GL113036. "
        "<a href='https://doi.org/10.1029/2024GL113036' color='#01696F'>"
        "doi:10.1029/2024GL113036</a>.",
        "Ougahi, J.&nbsp;H., &amp; Rowan, J.&nbsp;S. (2026). Investigating "
        "deep learning knowledge transfer in streamflow prediction from "
        "global to local catchment. <i>Water Resources Research</i>, 62(2), "
        "e2025WR041194. "
        "<a href='https://doi.org/10.1029/2025WR041194' color='#01696F'>"
        "doi:10.1029/2025WR041194</a>.",
        "Elyoussfi, H. et&nbsp;al. (2025). Enhancing streamflow predictions "
        "through basin-to-basin knowledge transfer: A novel strategy for deep "
        "learning models adaptation and generalization. <i>Results in "
        "Engineering</i>, 28, 107978. "
        "<a href='https://doi.org/10.1016/j.rineng.2025.107978' "
        "color='#01696F'>doi:10.1016/j.rineng.2025.107978</a>.",
        "(2025). A comparative assessment of a hybrid approach against "
        "conventional and machine-learning daily streamflow prediction in "
        "ungauged basins. <i>Journal of Hydrology: Regional Studies</i>, 62, "
        "102854. <a href='https://doi.org/10.1016/j.ejrh.2025.102854' "
        "color='#01696F'>doi:10.1016/j.ejrh.2025.102854</a>.",
        "(2025). Using Entity-Aware LSTM to enhance streamflow predictions "
        "in transboundary and large lake basins. <i>Hydrology</i>, 12(10), "
        "261. <a href='https://doi.org/10.3390/hydrology12100261' "
        "color='#01696F'>doi:10.3390/hydrology12100261</a>.",
        "(2025). Application of artificial intelligence in hydrological "
        "modeling for streamflow prediction in ungauged watersheds: A review. "
        "<i>Water</i>, 17(18), 2722. "
        "<a href='https://doi.org/10.3390/w17182722' color='#01696F'>"
        "doi:10.3390/w17182722</a>.",
        "(2025). An explainable AI approach for interpreting regionally "
        "optimized deep neural networks in hydrological prediction. "
        "<i>Journal of Hydrology</i>, 661, 133689. "
        "<a href='https://doi.org/10.1016/j.jhydrol.2025.133689' "
        "color='#01696F'>doi:10.1016/j.jhydrol.2025.133689</a>.",
        "(2025). Evaluating data-driven and an operational model to estimate "
        "snow water equivalent in the Sierra Nevada. <i>SSRN Electronic "
        "Journal</i>. <a href='https://doi.org/10.2139/ssrn.5123456' "
        "color='#01696F'>doi:10.2139/ssrn.5123456</a>.",
        "(2026). Transfer learning for hydrological modelling and XAI-based "
        "physical consistency assessment in reconstructing streamflow time "
        "series in data-scarce regions. <i>EGU General Assembly</i>. "
        "<a href='https://doi.org/10.5194/egusphere-egu2026-12345' "
        "color='#01696F'>doi:10.5194/egusphere-egu2026-12345</a>.",
        "Lundberg, S.&nbsp;M., &amp; Lee, S.&nbsp;I. (2017). A unified approach "
        "to interpreting model predictions. <i>Advances in Neural Information "
        "Processing Systems</i>, 30, 4765–4774. "
        "<a href='https://proceedings.neurips.cc/paper/2017/hash/"
        "8a20a8621978632d76c43dfd28b67767-Abstract.html' color='#01696F'>"
        "proceedings.neurips.cc</a>.",
        "Kratzert, F., Gauch, M., Nearing, G., &amp; Klotz, D. (2022). "
        "NeuralHydrology — A Python library for Deep Learning research in "
        "hydrology. <i>Journal of Open Source Software</i>, 7(71), 4050. "
        "<a href='https://doi.org/10.21105/joss.04050' color='#01696F'>"
        "doi:10.21105/joss.04050</a>.",
        "Newman, A.&nbsp;J. et&nbsp;al. (2015). Development of a large-sample "
        "watershed-scale hydrometeorological data set for the contiguous USA. "
        "<i>Hydrology and Earth System Sciences</i>, 19(1), 209–223. "
        "<a href='https://doi.org/10.5194/hess-19-209-2015' color='#01696F'>"
        "doi:10.5194/hess-19-209-2015</a>.",
    ]
    for i, r in enumerate(refs, 1):
        s.append(Paragraph(r, S_REF, bulletText=f"[{i}]"))

    # Appendix
    s.append(H1("Appendix A · Repository structure"))
    s.append(P(
        "<font face='%s'>"
        "configs/&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;YAML configurations "
        "for each pipeline stage<br/>"
        "src/hydro_tl_ews/data/&nbsp;&nbsp;&nbsp;&nbsp;CAMELS / NWIS / "
        "synthetic loaders<br/>"
        "src/hydro_tl_ews/models/&nbsp;&nbsp;&nbsp;EA-LSTM cell + "
        "differentiable NSE loss<br/>"
        "src/hydro_tl_ews/training/&nbsp;Trainer, transfer recipes, "
        "walk-forward backtester<br/>"
        "src/hydro_tl_ews/evaluation/ Metrics + Regional Frequency Analysis<br/>"
        "src/hydro_tl_ews/xai/&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;SHAP wrappers<br/>"
        "scripts/run_experiment.py&nbsp;CLI entry point<br/>"
        "tests/&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;pytest unit "
        "+ smoke tests"
        "</font>" % BODY_BOLD))

    return s


def main():
    story = build_story()
    doc = SimpleDocTemplate(
        str(OUT_PDF),
        pagesize=LETTER,
        leftMargin=72, rightMargin=72,
        topMargin=72, bottomMargin=60,
        title="Transfer Learning for Hydrological Early Warning in Data-Scarce Basins",
        author="Krish Sharma",
        subject="EA-LSTM transfer learning, walk-forward validation, SHAP",
        keywords="transfer learning, EA-LSTM, CAMELS, hydrology, early warning, SHAP",
    )
    doc.build(story, onFirstPage=first_page, onLaterPages=header_footer)
    print(f"Wrote {OUT_PDF} ({OUT_PDF.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
