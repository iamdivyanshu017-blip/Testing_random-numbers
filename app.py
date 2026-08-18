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
    "NIST SP800-22": {"color": "#6366F1", "min_recommended_bits": 100000},
    "IID (SP800-90B)": {"color": "#0EA5E9", "min_recommended_bits": 1000},
    "AIS-31": {"color": "#14B8A6", "min_recommended_bits": 20000},
    "Min-Entropy": {"color": "#F59E0B", "min_recommended_bits": 1000},
    "Dieharder": {"color": "#EC4899", "min_recommended_bits": None},
}


def load_module(name, filepath, extra_syspath=None):
    if extra_syspath and extra_syspath not in sys.path:
        sys.path.insert(0, extra_syspath)
    spec = importlib.util.spec_from_file_location(name, filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Data parsing — explicit binary vs decimal (no silent auto-detection).
# Decimal mode groups N numbers into one binary "key" at a time, producing
# multiple keys that get tested independently and aggregated.
# ---------------------------------------------------------------------------

def parse_binary_input(raw: str, bits_per_key: int = 0) -> tuple[list[str], list[str]]:
    """
    Binary mode: by default treats the whole cleaned sequence as one key
    (bits_per_key=0). If bits_per_key > 0, splits the sequence into that
    many bits per key, producing multiple keys. Trailing leftover bits
    that don't complete a full key are dropped (with a warning).
    """
    cleaned = "".join(c for c in raw if c in "01")
    if not cleaned:
        return [], []

    if bits_per_key <= 0:
        return [cleaned], []

    keys = []
    warnings = []
    for i in range(0, len(cleaned), bits_per_key):
        chunk = cleaned[i:i + bits_per_key]
        if len(chunk) < bits_per_key:
            warnings.append(
                f"Dropped trailing {len(chunk)} leftover bits "
                f"(needed {bits_per_key} to form a full key)."
            )
            continue
        keys.append(chunk)
    return keys, warnings


def parse_decimal_keys(raw: str, bit_width: int, numbers_per_key: int) -> tuple[list[str], list[str]]:
    """
    Parse whitespace/comma/newline-separated decimal numbers, convert each to
    a fixed-width binary value, and group every `numbers_per_key` of them into
    one concatenated binary key. A trailing incomplete group is dropped (with
    a warning) so every key is the same length.
    """
    tokens = re.split(r"[\s,]+", raw.strip())
    tokens = [t for t in tokens if t]

    max_val = 2 ** bit_width - 1
    bit_tokens = []
    warnings = []

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
                f"Value {n} exceeds {bit_width}-bit range (max {max_val}) — truncated to fit."
            )
            n = n & max_val
        bit_tokens.append(format(n, f"0{bit_width}b"))

    keys = []
    for i in range(0, len(bit_tokens), numbers_per_key):
        chunk = bit_tokens[i:i + numbers_per_key]
        if len(chunk) < numbers_per_key:
            warnings.append(
                f"Dropped trailing incomplete group of {len(chunk)} numbers "
                f"(needed {numbers_per_key} to form a full key)."
            )
            continue
        keys.append("".join(chunk))

    return keys, warnings


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


def mini_pill(status: str) -> str:
    style = VERDICT_STYLE.get(status, {"bg": "#F1F5F9", "fg": "#475569", "label": status})
    return f'<span style="background:{style["bg"]}; color:{style["fg"]}; border-radius:999px; padding:2px 10px; font-size:0.78rem; font-weight:700;">{style["label"]}</span>'


def pass_bar(passed, total, color="#6366F1", label="Pass rate"):
    if not total:
        st.caption("No countable results.")
        return
    pct = int(100 * passed / total)
    st.markdown(
        f"""
        <div style="margin:6px 0 14px 0;">
            <div style="display:flex; justify-content:space-between; font-size:0.85rem; color:#555; margin-bottom:4px;">
                <span>{label}</span><span>{passed}/{total} ({pct}%)</span>
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
# All (except Dieharder) take a single key's bit string and are called once
# per key by the aggregation loop below.
# ---------------------------------------------------------------------------

def run_nist(bits, alpha):
    """
    Run the local NIST SP 800-22 implementation on ONE sequence.

    Important:
    A single sequence produces multiple different statistical tests.
    Therefore, we do NOT interpret "39/40 tests passed" as a NIST
    proportion-of-sequences result.

    The proportion analysis belongs to multiple independent sequences
    for the SAME statistical test.
    """

    runner_path = os.path.join(NIST_DIR, "main_test_runner.py")
    nist_mod = load_module(
        "nist_runner",
        runner_path,
        extra_syspath=NIST_DIR
    )

    results = nist_mod.run_full_nist_suite(bits, alpha)

    rows = []

    pass_count = 0
    fail_count = 0
    skipped_count = 0
    error_count = 0

    for name, p_value, verdict in results:

        if isinstance(verdict, bool):

            if verdict:
                status = "PASS"
                pass_count += 1
            else:
                status = "FAIL"
                fail_count += 1

        else:
            status = str(verdict)

            if "SKIP" in status.upper():
                skipped_count += 1
            elif "ERROR" in status.upper():
                error_count += 1

        if isinstance(p_value, float):
            p_str = f"{p_value:.6f}"
        else:
            p_str = str(p_value)

        rows.append(
            {
                "Test": name,
                "P-Value": p_str,
                "Result": status,
            }
        )

    countable = pass_count + fail_count

    # ---------------------------------------------------------------
    # IMPORTANT:
    #
    # Do NOT do:
    #
    #     pass_count / countable >= 0.90
    #
    # because the rows are DIFFERENT statistical tests applied to
    # the SAME sequence.
    #
    # For a single sequence, we report the actual test outcomes.
    # ---------------------------------------------------------------

    if countable == 0:

        overall = "MIXED"

        subtitle = (
            "No countable NIST SP 800-22 tests were completed"
        )

    elif fail_count == 0 and error_count == 0:

        overall = "PASS"

        subtitle = (
            f"All {countable} countable NIST tests passed"
        )

    elif pass_count > 0:

        overall = "MIXED"

        subtitle = (
            f"{pass_count}/{countable} countable tests passed; "
            f"{fail_count} failed"
        )

    else:

        overall = "FAIL"

        subtitle = (
            f"0/{countable} countable tests passed"
        )

    # Add information about skipped/error tests.
    extra = []

    if skipped_count:
        extra.append(f"{skipped_count} skipped")

    if error_count:
        extra.append(f"{error_count} execution errors")

    if extra:
        subtitle += " (" + ", ".join(extra) + ")"

    def render_details():

        # -----------------------------------------------------------
        # DO NOT call this "NIST pass rate".
        #
        # It is simply a descriptive count of the individual tests
        # applied to this one sequence.
        # -----------------------------------------------------------

        if countable:
            pass_bar(
                pass_count,
                countable,
                SUITE_META["NIST SP800-22"]["color"],
                label="Individual test results"
            )

        st.info(
            "This percentage is descriptive only. "
            "NIST SP 800-22 proportion analysis is performed "
            "across multiple sequences for the same statistical test, "
            "not across different tests on one sequence."
        )

        st.dataframe(
            rows,
            use_container_width=True,
            hide_index=True
        )

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


def run_dieharder(selected_files=None, progress_callback=None, per_file_timeout=60):
    all_files = sorted(glob.glob(os.path.join(DIEHARD_DIR, "test_*.py")))
    test_files = [f for f in all_files if selected_files is None or os.path.basename(f) in selected_files]
    file_results = []
    pass_count, ran = 0, 0

    for i, filepath in enumerate(test_files):
        fname = os.path.basename(filepath)
        if progress_callback:
            progress_callback(i, len(test_files), fname)
        try:
            result = subprocess.run(
                [sys.executable, filepath],
                cwd=DIEHARD_DIR,
                capture_output=True,
                text=True,
                timeout=per_file_timeout,
                stdin=subprocess.DEVNULL,
            )
            output = result.stdout + result.stderr
        except subprocess.TimeoutExpired:
            output = f"Timed out after {per_file_timeout}s — this file is either slow or stuck waiting on input."
            file_results.append((fname, "TIMEOUT", output))
            continue

        match = re.search(r"RESULT:\s*(PASS(?:ED)?|FAIL(?:ED)?)", output, re.IGNORECASE)
        if match:
            ran += 1
            status = "PASS" if "PASS" in match.group(1).upper() else "FAIL"
            if status == "PASS":
                pass_count += 1
        else:
            status = "UNKNOWN"
        file_results.append((fname, status, output))

    if progress_callback:
        progress_callback(len(test_files), len(test_files), "Done")

    overall = "MIXED" if ran == 0 else ("PASS" if pass_count == ran else ("MIXED" if pass_count >= ran * 0.7 else "FAIL"))
    subtitle = f"{pass_count}/{ran} test files passed (using internal placeholder RNG, not your data)"

    def render_details():
        st.caption(
            "These 13 tests currently use their own internal placeholder RNG (numpy), "
            "not your data above — wiring real data into these files is a follow-up task."
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
}

# ---------------------------------------------------------------------------
# Page setup + global style
# ---------------------------------------------------------------------------
st.set_page_config(page_title="PUFSentinel", layout="wide")

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
            PUFSentinel
        </div>
        <div style="color:#777; font-size:0.95rem; margin-top:2px;">
            Randomness verification suite for GST-PUF key evaluation — load a key, run a suite, get a verdict.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

section_header("1", "Load your key", "#6366F1")

data_format = st.radio(
    "Data format",
    ["Binary (0s and 1s)", "Decimal numbers"],
    horizontal=True,
    help="Choose explicitly — auto-detecting this wrong has caused real bugs in this project before.",
)

bit_width = 32
numbers_per_key = 1
bits_per_key = 0
if data_format == "Decimal numbers":
    c1, c2 = st.columns(2)
    with c1:
        bit_width = st.selectbox(
            "Bits per number",
            [8, 16, 32, 64],
            index=2,
            help="Each decimal number becomes a fixed-width binary value (e.g. 8-bit for byte values 0-255).",
        )
    with c2:
        numbers_per_key = st.number_input(
            "Numbers per key",
            min_value=1,
            value=100,
            step=1,
            help=(
                "How many decimal numbers get grouped together to form one binary key. "
                "E.g. 100 numbers at 32 bits each = one 3200-bit key. Any leftover numbers "
                "that don't complete a full group are dropped."
            ),
        )
else:
    split_binary = st.checkbox(
        "Split this into multiple keys instead of treating it as one key",
        value=False,
    )
    if split_binary:
        bits_per_key = st.number_input(
            "Bits per key",
            min_value=1,
            value=20000,
            step=1000,
            help=(
                "Your pasted sequence gets split into keys of this many bits each. "
                "E.g. a 4,000,000-bit sequence at 20,000 bits/key = 200 keys. "
                "Leftover bits that don't complete a full key are dropped."
            ),
        )

col1, col2 = st.columns(2)
with col1:
    placeholder = "e.g. 110100101..." if data_format == "Binary (0s and 1s)" else "e.g. 109608821534377, 216968581907107, ..."
    pasted = st.text_area("Paste your PUF key/bitstream", height=120, placeholder=placeholder)
with col2:
    uploaded = st.file_uploader("...or upload a .txt file", type=["txt"])

raw_input = ""
if uploaded is not None:
    raw_input = uploaded.read().decode("utf-8", errors="ignore")
elif pasted.strip():
    raw_input = pasted

keys = []
parse_warnings = []
if raw_input:
    if data_format == "Binary (0s and 1s)":
        keys, parse_warnings = parse_binary_input(raw_input, bits_per_key)
    else:
        keys, parse_warnings = parse_decimal_keys(raw_input, bit_width, int(numbers_per_key))

if keys:
    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card("Keys generated", f"{len(keys):,}", "#22C55E")
    with c2:
        metric_card("Bits per key", f"{len(keys[0]):,}", "#22C55E")
    with c3:
        metric_card("Total bits", f"{sum(len(k) for k in keys):,}", "#22C55E")

    if parse_warnings:
        with st.expander(f"{len(parse_warnings)} warning(s) while parsing"):
            for w in parse_warnings:
                st.caption(w)
else:
    st.info("Paste your key or upload a file to continue.")

alpha = st.number_input("Significance level (alpha)", value=0.01, min_value=0.0001, max_value=0.5, step=0.01)

section_header("2", "Choose what to run", "#8B5CF6")

mode = st.radio("Run mode", ["Run all suites", "Run a single suite"], horizontal=True, label_visibility="collapsed")

all_suite_names = list(RUNNERS.keys()) + ["Dieharder"]
if mode == "Run a single suite":
    chosen = st.selectbox("Which suite?", all_suite_names)
    suites_to_run = [chosen]
else:
    suites_to_run = all_suite_names

if keys and len(keys) > 0:
    for suite_name in suites_to_run:
        min_req = SUITE_META.get(suite_name, {}).get("min_recommended_bits")
        if min_req and len(keys[0]) < min_req:
            st.warning(
                f"{suite_name} recommends at least {min_req:,} bits per key — "
                f"your keys are only {len(keys[0]):,} bits. Increase 'numbers per key' "
                f"or 'bits per number' above, or expect more SKIPped/invalid sub-tests. "
                f"(NIST in particular is designed for bulk RNG streams, not short "
                f"fixed-size keys — for best results, use 1,000,000+ bits, or treat "
                f"short-key NIST failures as expected rather than a bug.)"
            )

dieharder_selected = []
dieharder_timeout = 60
if "Dieharder" in suites_to_run:
    with st.expander("Dieharder options (13 tests can take a few minutes total)"):
        all_dh_files = sorted(os.path.basename(f) for f in glob.glob(os.path.join(DIEHARD_DIR, "test_*.py")))
        dieharder_selected = st.multiselect(
            "Which Dieharder tests to run? (leave empty to run all)",
            all_dh_files,
        )
        dieharder_timeout = st.number_input(
            "Per-file timeout (seconds)",
            min_value=5, value=60, step=5,
            help="If a single test takes longer than this, it's marked TIMEOUT and the rest continue.",
        )

run_clicked = st.button("Run", type="primary", disabled=not keys, use_container_width=True)

section_header("3", "Results", "#EC4899")

if not keys:
    st.info("Waiting for a key...")
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

            if suite_name == "Dieharder":
                progress_bar = st.progress(0)
                status_text = st.empty()

                def _update_progress(done, total, fname):
                    progress_bar.progress(done / total if total else 0)
                    status_text.caption(f"Running {fname}  ({done}/{total})")

                try:
                    overall, subtitle, render_details = run_dieharder(
                        selected_files=dieharder_selected or None,
                        progress_callback=_update_progress,
                        per_file_timeout=dieharder_timeout,
                    )
                except Exception as e:
                    st.error(f"Could not run this suite: {e}")
                    continue
                progress_bar.empty()
                status_text.empty()
                verdict_pill(overall, subtitle)
                with st.expander("View test report details"):
                    render_details()
                continue

            # --- Per-key aggregation for all other suites ---
            per_key_results = []
            with st.spinner(f"Running {suite_name} on {len(keys)} key(s)..."):
                for idx, key_bits in enumerate(keys):
                    try:
                        k_overall, k_subtitle, k_render = RUNNERS[suite_name](key_bits, alpha)
                    except Exception as e:
                        k_overall, k_subtitle, k_render = "MIXED", f"Error: {e}", (lambda: st.error(str(e)))
                    per_key_results.append((idx, k_overall, k_subtitle, k_render))

                        # -------------------------------------------------------
            # Per-key results
            # -------------------------------------------------------

            per_key_results = []

            with st.spinner(
                f"Running {suite_name} on {len(keys)} key(s)..."
            ):

                for idx, key_bits in enumerate(keys):

                    try:

                        k_overall, k_subtitle, k_render = RUNNERS[
                            suite_name
                        ](
                            key_bits,
                            alpha
                        )

                    except Exception as e:

                        k_overall = "MIXED"
                        k_subtitle = f"Error: {e}"

                        k_render = (
                            lambda e=e: st.error(str(e))
                        )

                    per_key_results.append(
                        (
                            idx,
                            k_overall,
                            k_subtitle,
                            k_render
                        )
                    )

            # -------------------------------------------------------
            # IMPORTANT FOR NIST:
            #
            # If there is only ONE key/sequence, do NOT say:
            #
            #     39/40 keys passed
            #
            # because there is only one sequence.
            #
            # If multiple keys are supplied, each key is a sequence.
            # -------------------------------------------------------

            total = len(per_key_results)

            if total == 0:

                agg_status = "MIXED"

                agg_subtitle = (
                    "No sequences were available for testing"
                )

            elif total == 1:

                # One sequence cannot produce a proportion-of-sequences
                # analysis.

                _, key_status, key_subtitle, _ = per_key_results[0]

                agg_status = key_status

                agg_subtitle = (
                    "Single sequence: individual NIST test results shown; "
                    "proportion analysis is not applicable"
                )

            else:

                            # ---------------------------------------------------------------
             # Aggregate results across keys/sequences
             # ---------------------------------------------------------------

             total = len(per_key_results)

             # Always initialize these values.
             pass_count = sum(
                1
                for _, overall, _, _ in per_key_results
                if overall == "PASS"
             ) 

             fail_count = sum(
                1
                for _, overall, _, _ in per_key_results
                if overall == "FAIL"
             )

             mixed_count = sum(
                1
                for _, overall, _, _ in per_key_results
                if overall == "MIXED"
             ) 

             # ---------------------------------------------------------------
             # Determine overall status
             # ---------------------------------------------------------------

            if total == 0:

                agg_status = "MIXED"

                agg_subtitle = (
                    "No sequences were available for testing"
                )

            elif total == 1:

                # Only one sequence/key was tested.
                #
                # Do NOT perform a proportion-of-sequences analysis.
                # Simply use the result of that one sequence.

                agg_status = per_key_results[0][1]

                agg_subtitle = (
                    "Single sequence: individual NIST test results shown; "
                    "proportion analysis is not applicable"
                )

            else:

                # Multiple independent sequences/keys.

                if pass_count == total:

                    agg_status = "PASS"

                elif fail_count == total:

                    agg_status = "FAIL"

                else:

                    agg_status = "MIXED"

                agg_subtitle = (
                    f"{pass_count}/{total} sequences passed overall; "
                    f"{fail_count} failed; "
                    f"{mixed_count} mixed/inconclusive"
                )

            # ---------------------------------------------------------------
            # Display overall verdict
            # ---------------------------------------------------------------

            verdict_pill(
                agg_status,
                agg_subtitle
            )

            # ---------------------------------------------------------------
            # Display detailed results
            # ---------------------------------------------------------------

            with st.expander("View test report details"):

                if total > 1:

                    pass_bar(
                        pass_count,
                        total,
                        meta["color"],
                        label="Sequences passed overall"
                    )

                    st.info(
                        "For NIST SP 800-22, proportion analysis should be "
                        "performed separately for each statistical test across "
                        "multiple independent sequences. This bar summarizes "
                        "whole-sequence outcomes only."
                    )

                else:

                    st.info(
                        "Only one sequence was tested. NIST proportion analysis "
                        "is not applicable to a single sequence."
                    )

                # -----------------------------------------------------------
                # Show each key/sequence
                # -----------------------------------------------------------

                for (
                    idx,
                    k_overall,
                    k_subtitle,
                    k_render
                ) in per_key_results:

                    label = VERDICT_STYLE.get(
                        k_overall,
                        {}
                    ).get(
                        "label",
                        k_overall
                    )

                    with st.expander(
                        f"Key {idx + 1} — {label}"
                    ):

                        st.caption(k_subtitle)

                        k_render()
else:
    st.info("Click Run above to see results here.")
