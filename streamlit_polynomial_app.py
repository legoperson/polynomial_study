# Streamlit Polynomial Plotter
# -------------------------------------------------------------
# Features
# - Input maximum degree (0–20)
# - Enter coefficients either via fields or by pasting comma-separated values
# - Plot y(x) over [-100, 100] with step 0.1
# - Show the polynomial in descending powers (LaTeX)
# - Optional y-axis limits to avoid extreme scaling
# - Download CSV of sampled (x, y)
# - NEW: Keep all previously plotted curves, and a button to clear all curves
#
# Run locally:
#   pip install streamlit numpy matplotlib pandas
#   streamlit run streamlit_polynomial_app.py

import io
from typing import List

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Polynomial Plotter", page_icon="➗", layout="wide")

# -------------------------
# Global grid & session state
# -------------------------

# x-range constants
X_MIN, X_MAX, STEP = -100.0, 100.0, 0.1
X_GRID = np.arange(X_MIN, X_MAX + 1e-9, STEP, dtype=float)

# Store all plotted curves and the last plotted curve
if "curves" not in st.session_state:
    st.session_state["curves"] = []         # list of {"coeff_desc", "y", "label"}
if "last_curve" not in st.session_state:
    st.session_state["last_curve"] = None   # same structure as above


# -------------------------
# Helpers
# -------------------------

def format_poly_latex(coeff_desc: List[float]) -> str:
    """Return a LaTeX string of the polynomial in descending powers.
    coeff_desc is a list [a_n, a_{n-1}, ..., a_0]."""
    terms = []
    n = len(coeff_desc) - 1
    for i, a in enumerate(coeff_desc):
        p = n - i
        if a == 0:
            continue
        # sign and abs
        sign = "+" if (a > 0 and len(terms) > 0) else ("-" if a < 0 else "")
        aval = abs(a)
        # coefficient string
        if p == 0:
            coeff_str = f"{aval:g}"
        elif aval == 1:
            coeff_str = ""
        else:
            coeff_str = f"{aval:g}"
        # power string
        if p == 0:
            power_str = ""
        elif p == 1:
            power_str = "x"
        else:
            power_str = f"x^{p}"
        # multiply symbol if needed
        mult = "" if (coeff_str == "" or power_str == "") else " \\cdot "
        term = f"{sign} {coeff_str}{mult}{power_str}".strip()
        # Clean up leading plus
        if len(terms) == 0 and sign == "+":
            term = term[2:] if term.startswith("+ ") else term
        terms.append(term)
    if not terms:
        return "0"
    return " ".join(terms)


def parse_coeffs_from_text(txt: str, degree: int) -> List[float]:
    """Parse comma/space separated coefficients. Accept either
    descending (a_n ... a_0) or ascending (a_0 ... a_n) if length matches.
    If provided length < degree+1, pad with zeros on the *right* for ascending,
    or on the *left* for descending (we try to guess orientation)."""
    if not txt.strip():
        return [0.0] * (degree + 1)
    raw = [t for t in txt.replace("\n", ",").replace(" ", ",").split(",") if t.strip() != ""]
    vals = []
    for r in raw:
        try:
            vals.append(float(r))
        except ValueError:
            # Ignore non-numeric tokens
            pass
    if not vals:
        return [0.0] * (degree + 1)
    # If exact length, try to auto-detect orientation by comparing magnitudes
    if len(vals) == degree + 1:
        # Heuristic: if last term magnitude > first, assume ascending (common when typing a0,...,an)
        ascending_guess = abs(vals[-1]) < abs(vals[0])
        coeff_desc = vals if ascending_guess else list(reversed(vals))
        # Actually let's do a simpler, robust rule: assume user pasted ascending a0..an
        coeff_desc = list(reversed(vals))
        return coeff_desc
    # If shorter, decide padding orientation by defaulting to ascending input
    if len(vals) < degree + 1:
        vals = vals + [0.0] * ((degree + 1) - len(vals))  # pad as ascending
        return list(reversed(vals))
    # If longer than needed, trim
    vals = vals[: degree + 1]
    return list(reversed(vals))


def compute_poly_y(coeff_desc: List[float], x: np.ndarray) -> np.ndarray:
    # numpy.polyval expects descending coefficients
    c = np.array(coeff_desc, dtype=float)
    with np.errstate(over="ignore", invalid="ignore"):  # avoid warnings for huge values
        y = np.polyval(c, x)
    return y


# -------------------------
# UI
# -------------------------

st.title("Polynomial Plotter (Streamlit)")
st.caption("Input max degree and coefficients to plot y over [-100, 100] with step 0.1.")

with st.sidebar:
    st.header("Settings")
    degree = st.number_input("Maximum degree n", min_value=0, max_value=20, value=2, step=1)

    input_mode = st.radio(
        "Coefficient input mode",
        options=["Fields (descending a_n..a_0)", "Paste (comma-separated)"],
        index=0,
        help=(
            "Descending means coefficients are ordered a_n, a_{n-1}, …, a_0.\n"
            "If you paste, you can input ascending a_0,…,a_n; the app will auto-handle."
        ),
    )

    if input_mode.startswith("Fields"):
        st.markdown("**Enter coefficients in descending powers (a_n → a_0):**")
        coeff_desc: List[float] = []
        for p in range(int(degree), -1, -1):
            label = f"aₙ for x^{p}" if p > 1 else ("aₙ for x" if p == 1 else "aₙ constant")
            coeff = st.number_input(
                label,
                key=f"coeff_{p}",
                value=0.0,
                format="%g",
            )
            coeff_desc.append(coeff)
    else:
        placeholder = (
            "Example ascending a_0,…,a_n: 4, 5, 2  (for 2x^2 + 5x + 4)\n"
            "Example descending a_n,…,a_0: 2, 5, 4"
        )
        pasted = st.text_area(
            "Paste coefficients (comma/space separated)",
            value="",
            height=100,
            help=placeholder,
        )
        coeff_desc = parse_coeffs_from_text(pasted, int(degree))

    # Plot controls
    st.divider()
    st.subheader("Plot Controls")
    x_min, x_max = X_MIN, X_MAX
    step = STEP
    custom_ylim = st.checkbox("Set custom y-limits (avoid extreme scaling)", value=False)
    y_min, y_max = None, None
    if custom_ylim:
        y_min = st.number_input("y min", value=-1000.0, format="%g")
        y_max = st.number_input("y max", value=1000.0, format="%g")
        if y_max <= y_min:
            st.warning("y max must be > y min; using auto-scale instead.")
            custom_ylim = False

    st.divider()
    # New controls: add curve / clear canvas
    plot_clicked = st.button("Plot current polynomial (add curve)", type="primary")
    clear_clicked = st.button("Clear all curves")

    if plot_clicked:
        # Compute y for the current coefficients and store as a new curve
        y_new = compute_poly_y(coeff_desc, X_GRID)
        curve = {
            "coeff_desc": coeff_desc.copy(),
            "y": y_new,
            "label": format_poly_latex(coeff_desc),
        }
        st.session_state["curves"].append(curve)
        st.session_state["last_curve"] = curve

    if clear_clicked:
        st.session_state["curves"] = []
        st.session_state["last_curve"] = None


# Body
col1, col2 = st.columns([3, 2], gap="large")

with col1:
    st.subheader("Polynomial")
    # Show current polynomial (according to sidebar inputs)
    latex_str = format_poly_latex(coeff_desc)
    st.latex(r"y(x) = " + latex_str)

    x = X_GRID

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=140)

    if st.session_state["curves"]:
        # Plot all stored curves so previous lines stay visible
        for idx, curve in enumerate(st.session_state["curves"], start=1):
            label = curve.get("label", f"Curve {idx}")
            ax.plot(x, curve["y"], label=label)
        ax.legend()
        ax.set_title("y = polynomial(x)  (multiple curves)")
    else:
        # No stored curves yet – show a preview of the current polynomial
        y_preview = compute_poly_y(coeff_desc, x)
        ax.plot(x, y_preview, linestyle="--")
        ax.set_title("Preview of current polynomial (not yet added)")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.3)
    if custom_ylim and (y_min is not None and y_max is not None):
        try:
            ax.set_ylim([y_min, y_max])
        except Exception:
            pass
    st.pyplot(fig, clear_figure=True)

with col2:
    st.subheader("Data & Export")
    st.caption("Sampled at step = 0.1 in [-100, 100].")

    if st.session_state["last_curve"] is not None:
        y_vals = st.session_state["last_curve"]["y"]
        df = pd.DataFrame({"x": X_GRID, "y": y_vals})

        finite_mask = np.isfinite(y_vals)
        if finite_mask.any():
            ymin = float(np.min(y_vals[finite_mask]))
            ymax = float(np.max(y_vals[finite_mask]))
            st.metric("min(y) of last curve", f"{ymin:g}")
            st.metric("max(y) of last curve", f"{ymax:g}")
        else:
            st.info(
                "All values of the last plotted curve are non-finite (overflow). "
                "Try reducing degree/coefficients or set y-limits."
            )

        csv_buf = io.StringIO()
        df.to_csv(csv_buf, index=False)
        st.download_button(
            "Download CSV of last curve",
            data=csv_buf.getvalue(),
            file_name
