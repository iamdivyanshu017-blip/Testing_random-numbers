"""
Unified PUF/TRNG Randomness Testing Dashboard
-----------------------------------------------
Save this file directly inside your "Akshaya Jha- Intern Work" folder
(the one containing "AIS Test Suites", "Diehard Test Suite",
"IID Test Suite", "Minimum Entropy Calculator", "NIST_Random_testing").

Run with:
    pip install streamlit numpy scipy matplotlib
    streamlit run app.py
"""

import os
import re
import sys
import io
import glob
import subprocess
import importlib.util
import contextlib

import streamlit as st
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

AIS_DIR = os.path.join(BASE_DIR, "AIS Test Suites")
DIEHARD_DIR = os.path.join(BASE_DIR, "Diehard Test Suite")
IID_DIR = os.path.join(BASE_DIR, "IID Test Suite")
ENTROPY_DIR = os.path.join(BASE_DIR, "Minimum Entropy Calculator")
NIST_DIR = os.path.join(BASE_DIR, "NIST_Random_testing")

SUITE_META = {
    "NIST SP800-22": {"color": "#6366F1"},
    "IID (SP800-90B)": {"color": "#0EA5E9"},
    "AIS-31": {"color": "#14B8A6"},
    "Min-Entropy": {"color": "#F59E0B"},
    "Dieharder": {"color": "#EC4899"},
}


def load_module(name, filepath, extra_syspath=None):
    if extra_syspath and extra_syspath not in sys.path:
        sys.path.insert(0, extra_syspath)
    spec = importlib.util.spec_from_file_location(name, filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Data parsing — explicit binary vs decimal (no silent auto-detection,
# since guessing wrong here has already caused real bugs in this project)
# ---------------------------------------------------------------------------

def parse_binary_input(raw: str) -> str:
    """Keep only 0/1 characters."""
    return "".join(c for c in raw if c in "01")


def parse_decimal_input(raw: str, bit_width: int) -> tuple[str, list[str]]:
    """
    Parse whitespace/comma/newline-separated decimal numbers and convert
    each to a fixed-width binary string, then concatenate.
    Returns (bit_string, list_of_warnings).
    """
    tokens = re.split(r"[\s,]+", raw.strip())
    tokens = [t for t in tokens if t]

    bits_out = []
    warnings = []
    max_val = 2 ** bit_width - 1

    for tok in tokens:
        try:
            n = int(tok)
        except ValueError:
            warnings.append(f"Skipped non-numeric token: '{tok}'")
            continue
        if n < 0:
            warnings.append(f"Skipped negative number: {n}")
            continue
        if n > max_val:
            warnings.append(
                f"Value {n} exceeds {bit_width}-bit range (max {max_val}) — "
                f"increase 'bits per number' to represent it fully."
            )
            n = n & max_val  # truncate to fit, but flag it
        bits_out.append(format(n, f"0{bit_width}b"))

    return "".join(bits_out), warnings


# ---------------------------------------------------------------------------
# Visual helpers
# ---------------------------------------------------------------------------

VERDICT_STYLE = {
    "PASS": {"bg": "#DCFCE7", "fg": "#15803D", "label": "PASS"},
    "FAIL": {"bg": "#FEE2E2", "fg": "#B91C1C", "label": "FAIL"},
    "MIXED": {"bg": "#FEF3C7", "fg": "#B45309", "label": "MIXED / INCONCLUSIVE"},
}


def verdict_pill(status: str, subtitle: str = ""):
    style = VERDICT_STYLE.get(status)
    if style:
        st.markdown(
            f"""
            <div style="
                background:{style['bg']}; color:{style['fg']};
                border-radius:14px; padding:16px 20px; margin:8px 0 4px 0;
                font-weight:700; font-size:1.05rem;">
                <div>FINAL VERDICT: {style['label']}</div>
                {f'<div style="font-weight:400; font-size:0.85rem; opacity:0.85; margin-top:2px;">{subtitle}</div>' if subtitle else ''}
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div style="
                background:#EEF2FF; color:#4338CA; border-radius:14px;
                padding:16px 20px; margin:8px 0 4px 0; font-weight:700;">
                {status}
            </div>
            """,
            unsafe_allow_html=True,
        )


def pass_bar(passed, total, color="#6366F1"):
    if not total:
        st.caption("No countable tests ran.")
        return
    pct = int(100 * passed / total)
    st.markdown(
        f"""
        <div style="margin:6px 0 14px 0;">
            <div style="display:flex; justify-content:space-between; font-size:0.85rem; color:#555; margin-bottom:4px;">
                <span>Pass rate</span><span>{passed}/{total} ({pct}%)</span>
            </div>
            <div style="background:#EEE; border-radius:999px; height:10px; overflow:hidden;">
                <div style="width:{pct}%; background:{color}; height:100%; border-radius:999px;"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_card(label, value, color="#6366F1"):
    st.markdown(
        f"""
        <div style="
            background:white; border:1px solid #EEE; border-left:5px solid {color};
            border-radius:12px; padding:14px 16px; text-align:left;">
            <div style="font-size:0.78rem; color:#888; font-weight:600; text-transform:uppercase; letter-spacing:0.03em;">{label}</div>
            <div style="font-size:1.5rem; font-weight:700; color:#222; margin-top:2px;">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_header(number, text, color="#6366F1"):
    st.markdown(
        f"""
        <div style="display:flex; align-items:center; gap:12px; margin:22px 0 8px 0;">
            <div style="background:{color}; color:white; border-radius:8px; width:26px; height:26px;
                        display:flex; align-items:center; justify-content:center; font-size:0.85rem; font-weight:700;">{number}</div>
            <div style="font-size:1.15rem; font-weight:700; color:#222;">{text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Suite runners — each returns (verdict_status, subtitle, detail_renderer_function)
# ---------------------------------------------------------------------------

def run_nist(bits, alpha):
    runner_path = os.path.join(NIST_DIR, "main_test_runner.py")
    nist_mod = load_module("nist_runner", runner_path, extra_syspath=NIST_DIR)
    results = nist_mod.run_full_nist_suite(bits, alpha)

    rows = []
    pass_count, countable = 0, 0
    for name, p_value, verdict in results:
        if isinstance(verdict, str):
            status = verdict
        else:
            status = "PASS" if verdict else "FAIL"
            countable += 1
            if verdict:
                pass_count += 1
        p_str = f"{p_value:.6f}" if isinstance(p_value, float) else str(p_value)
        rows.append({"Test": name, "P-Value": p_str, "Result": status})

    overall = "MIXED" if countable == 0 else ("PASS" if pass_count == countable else "FAIL")
    subtitle = f"{pass_count}/{countable} sub-tests passed" if countable else "No countable results"

    def render_details():
        pass_bar(pass_count, countable, SUITE_META["NIST SP800-22"]["color"])
        st.dataframe(rows, use_container_width=True, hide_index=True)

    return overall, subtitle, render_details


def run_iid(bits):
    main_path = os.path.join(IID_DIR, "main.py")
    iid_mod = load_module("iid_main", main_path, extra_syspath=IID_DIR)

    data = np.array([int(b) for b in bits], dtype=int)
    if len(np.unique(data)) <= 2:
        data = iid_mod.pack_bits_to_bytes(data)

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        iid_mod.run_all_tests(data, verbose=False)
    output_text = buf.getvalue()

    if "IID ASSUMPTION VALIDATED" in output_text:
        overall = "PASS"
    elif "IID ASSUMPTION REJECTED" in output_text:
        overall = "FAIL"
    else:
        overall = "MIXED"
    subtitle = "NIST SP800-90B IID battery (13 sub-tests)"

    def render_details():
        st.text(output_text)

    return overall, subtitle, render_details


def run_ais(bits):
    ais_path = os.path.join(AIS_DIR, "ais.py")
    ais_mod = load_module("ais_mod", ais_path)

    suite = [
        ("T1 Monobit", ais_mod.test_t1_monobit, 20000),
        ("T2 Poker", ais_mod.test_t2_poker, 20000),
        ("T3 Runs", ais_mod.test_t3_runs, 20000),
        ("T4 Long Run", ais_mod.test_t4_long_run, 20000),
        ("T5 Autocorr", ais_mod.test_t5_autocorr, 20000),
        ("T6 Uniform", ais_mod.test_t6_uniform, 100000),
        ("T7 Homogen", ais_mod.test_t7_homogeneity, 200000),
        ("T8 Entropy", ais_mod.test_t8_entropy, 225280),
        ("T0 Disjoint", ais_mod.test_t0_disjointness, 3145728),
    ]

    rows = []
    L = len(bits)
    ran, passed_count = 0, 0
    for name, func, req in suite:
        if L >= req:
            passed, info = func(bits)
            ran += 1
            if passed:
                passed_count += 1
            rows.append({"Test": name, "Result": "PASS" if passed else "FAIL", "Details": str(info)})
        else:
            rows.append({"Test": name, "Result": "SKIP", "Details": f"Needs {req} bits, have {L}"})

    overall = "MIXED" if ran == 0 else ("PASS" if passed_count == ran else "FAIL")
    subtitle = f"{ran}/9 tests had enough data to run" if ran else "Not enough data for any AIS-31 test"

    def render_details():
        pass_bar(passed_count, ran, SUITE_META["AIS-31"]["color"])
        st.dataframe(rows, use_container_width=True, hide_index=True)

    return overall, subtitle, render_details


def run_entropy(bits):
    entropy_path = os.path.join(ENTROPY_DIR, "min.py")
    entropy_mod = load_module("entropy_mod", entropy_path)

    h_mcv, p_mcv = entropy_mod.calculate_mcv_estimate(bits)
    h_markov, p_markov = entropy_mod.calculate_markov_estimate(bits)
    final_h_min = min(h_mcv, h_markov)

    overall = "PASS" if final_h_min >= 0.9 else ("MIXED" if final_h_min >= 0.75 else "FAIL")
    subtitle = f"Final min-entropy: {final_h_min:.4f} bits/bit (ideal = 1.0)"

    def render_details():
        c1, c2, c3 = st.columns(3)
        with c1:
            metric_card("MCV Entropy", f"{h_mcv:.4f} bits", SUITE_META["Min-Entropy"]["color"])
        with c2:
            metric_card("Markov Entropy", f"{h_markov:.4f} bits", SUITE_META["Min-Entropy"]["color"])
        with c3:
            metric_card("Final H-min", f"{final_h_min:.4f} bits/bit", SUITE_META["Min-Entropy"]["color"])
        st.caption(
            "Covers 2 of the ~10 estimators used in the full NIST SP800-90B min-entropy tool "
            "(MCV + Markov only) — treat this as an optimistic estimate vs. the official tool."
        )

    return overall, subtitle, render_details


def run_dieharder():
    test_files = sorted(glob.glob(os.path.join(DIEHARD_DIR, "test_*.py")))
    file_results = []
    pass_count, ran = 0, 0

    for filepath in test_files:
        fname = os.path.basename(filepath)
        result = subprocess.run(
            [sys.executable, filepath],
            cwd=DIEHARD_DIR,
            capture_output=True,
            text=True,
            timeout=300,
        )
        output = result.stdout + result.stderr
        match = re.search(r"RESULT:\s*(PASS(?:ED)?|FAIL(?:ED)?)", output, re.IGNORECASE)
        if match:
            ran += 1
            status = "PASS" if "PASS" in match.group(1).upper() else "FAIL"
            if status == "PASS":
                pass_count += 1
        else:
            status = "UNKNOWN"
        file_results.append((fname, status, output))

    overall = "MIXED" if ran == 0 else ("PASS" if pass_count == ran else "FAIL")
    subtitle = f"{pass_count}/{ran} test files passed (using internal placeholder RNG, not your data)"

    def render_details():
        st.caption(
            "These 13 tests currently use their own internal placeholder RNG (numpy), "
            "not your pasted data above — wiring real data into these files is a follow-up task."
        )
        pass_bar(pass_count, ran, SUITE_META["Dieharder"]["color"])
        for fname, status, output in file_results:
            with st.expander(f"{fname}  —  {status}"):
                st.text(output)

    return overall, subtitle, render_details


RUNNERS = {
    "NIST SP800-22": lambda bits, alpha: run_nist(bits, alpha),
    "IID (SP800-90B)": lambda bits, alpha: run_iid(bits),
    "AIS-31": lambda bits, alpha: run_ais(bits),
    "Min-Entropy": lambda bits, alpha: run_entropy(bits),
    "Dieharder": lambda bits, alpha: run_dieharder(),
}

# ---------------------------------------------------------------------------
# Page setup + global style
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Randomness Test Dashboard", layout="wide")

st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
        .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1100px; }
        .stButton > button {
            border-radius: 12px; font-weight: 700; padding: 0.6rem 1.2rem;
            border: none; background: linear-gradient(135deg, #6366F1, #8B5CF6);
            color: white; transition: transform 0.15s ease, box-shadow 0.15s ease;
        }
        .stButton > button:hover {
            transform: translateY(-1px); box-shadow: 0 6px 16px rgba(99,102,241,0.35);
            color: white;
        }
        .stTextArea textarea, .stFileUploader, .stNumberInput input {
            border-radius: 10px !important;
        }
        div[data-testid="stExpander"] {
            border-radius: 12px !important; border: 1px solid #EEE !important;
        }
        div[data-testid="stVerticalBlockBorderWrapper"] {
            border-radius: 16px !important; border: 1px solid #EEE !important;
            box-shadow: 0 2px 10px rgba(0,0,0,0.03);
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div style="padding:6px 0 0 0;">
        <div style="font-size:2rem; font-weight:800;
                    background: linear-gradient(135deg, #6366F1, #EC4899);
                    -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            PUF / TRNG Randomness Test Dashboard
        </div>
        <div style="color:#777; font-size:0.95rem; margin-top:2px;">
            Paste or upload your data, choose what to run, and get a clean verdict for each suite.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

section_header("1", "Provide your data", "#6366F1")

data_format = st.radio(
    "Data format",
    ["Binary (0s and 1s)", "Decimal numbers"],
    horizontal=True,
    help="Choose explicitly — auto-detecting this wrong has caused real bugs in this project before.",
)

bit_width = 32
if data_format == "Decimal numbers":
    bit_width = st.selectbox(
        "Bits per number",
        [8, 16, 32, 64],
        index=2,
        help=(
            "Each decimal number is converted to a fixed-width binary value and "
            "concatenated. Pick the width that matches how your data was generated "
            "(e.g. 8 for byte values 0-255, 32 for typical random integers)."
        ),
    )

col1, col2 = st.columns(2)
with col1:
    placeholder = "e.g. 110100101..." if data_format == "Binary (0s and 1s)" else "e.g. 109608821534377, 216968581907107, ..."
    pasted = st.text_area("Paste your data", height=120, placeholder=placeholder)
with col2:
    uploaded = st.file_uploader("...or upload a .txt file", type=["txt"])

raw_input = ""
if uploaded is not None:
    raw_input = uploaded.read().decode("utf-8", errors="ignore")
elif pasted.strip():
    raw_input = pasted

bits = ""
parse_warnings = []
if raw_input:
    if data_format == "Binary (0s and 1s)":
        bits = parse_binary_input(raw_input)
    else:
        bits, parse_warnings = parse_decimal_input(raw_input, bit_width)

if bits:
    c1, c2 = st.columns(2)
    with c1:
        metric_card("Bits loaded", f"{len(bits):,}", "#22C55E")
    with c2:
        if data_format == "Decimal numbers":
            metric_card("Numbers parsed", f"{len(bits) // bit_width:,}", "#22C55E")
    if parse_warnings:
        with st.expander(f"⚠ {len(parse_warnings)} warning(s) while parsing"):
            for w in parse_warnings:
                st.caption(w)
else:
    st.info("Paste data or upload a file to continue.")

alpha = st.number_input("Significance level (alpha)", value=0.01, min_value=0.0001, max_value=0.5, step=0.01)

section_header("2", "Choose what to run", "#8B5CF6")

mode = st.radio("Run mode", ["Run all suites", "Run a single suite"], horizontal=True, label_visibility="collapsed")

if mode == "Run a single suite":
    chosen = st.selectbox("Which suite?", list(RUNNERS.keys()))
    suites_to_run = [chosen]
else:
    suites_to_run = list(RUNNERS.keys())

run_clicked = st.button("Run", type="primary", disabled=not bits, use_container_width=True)

section_header("3", "Results", "#EC4899")

if not bits:
    st.info("Waiting for data...")
elif run_clicked:
    for suite_name in suites_to_run:
        meta = SUITE_META[suite_name]
        with st.container(border=True):
            st.markdown(
                f"""
                <div style="display:flex; align-items:center; gap:10px; padding:4px 0 0 4px;">
                    <div style="width:10px; height:10px; border-radius:999px; background:{meta['color']};"></div>
                    <span style="font-size:1.15rem; font-weight:700; color:#222;">{suite_name}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            with st.spinner(f"Running {suite_name}..."):
                try:
                    overall, subtitle, render_details = RUNNERS[suite_name](bits, alpha)
                except Exception as e:
                    st.error(f"Could not run this suite: {e}")
                    continue

            if overall in VERDICT_STYLE:
                verdict_pill(overall, subtitle)
            else:
                verdict_pill("INFO", overall)

            with st.expander("View test report details"):
                render_details()
else:
    st.info("Click Run above to see results here.")
