"""
Ground-truth click probability formula.  Used ONLY to generate the synthetic
dataset.  Trained ML models do not see this function — only (X, y) pairs.

Decomposition
-------------
    p(click) = p_attractive * penalty * visibility

    p_attractive = sigmoid(linear combination of accessibility/UX signals)
    penalty      = product of multiplicative penalties (each in (0, 1])
    visibility   = p(seen) based on vertical button position

Formula changes vs v1
---------------------
1. Colour harmony (Itten 1961; Ou & Luo, Color Res. Appl. 2006)
   Added f_colour_harmony() as a multiplicative penalty.
   - Same or very similar hue → near-zero score (text invisible on bg).
   - Complementary (≈180 °) → score 1.0; triadic, split-complementary → 0.85–0.9.
   - Greyscale pairs (low saturation on both sides) fall back to 0.7 (neutral).

2. Margin/padding balance (Lidwell et al., "Universal Principles of Design", 2010;
   Google Material Design 2024)
   Added f_margin_balance(): penalises uneven horizontal vs vertical text
   clearance inside the button.  Derivation uses (whitespace_ratio * btn_w) as
   approximate per-side horizontal clearance and (btn_h − font_size)/2 as
   vertical clearance.

3. Improved vertical-position model (replaces simple exp(-lam * scroll))
   f_position_decay() now has a sweet-spot at scroll_to_button = 0.15:
   - scroll = 0   → 0.85  (top of page, above value proposition; slight penalty)
   - scroll = 0.15 → 1.00  (optimal: just below the value proposition / hero text)
   - scroll > 0.15 → exponential decay with lam = 3.0 (stronger than old lam = 2.0)
   References:
   - Chartbeat (2014, 2 B sessions): engagement drops ~60 % per page-fold.
   - Nielsen Norman Group / Fessenden (2018): CTAs above the fold convert up to
     47 % more; optimal CTA zone is 10–30 % down the page.

4. lambda updated from 2.0 → 3.0 (steeper scroll penalty) to align with
   Chartbeat's ~60 % drop per fold.

Anchors (unchanged from v1)
---------------------------
    contrast     → WCAG 2.1 (W3C, 2018)
    size         → Baymard 2024, Apple HIG, Google Material
    text_quality → KISSmetrics, CXL
    time         → Infolinks (1 T impressions)
    whitespace   → VWO / Open Mile (+232 %)
    scroll decay → Chartbeat (2 B sessions), NN/g (2018)
"""

from __future__ import annotations
import colorsys
import numpy as np
from typing import Dict, Optional, Tuple


# ---------------------------------------------------------------------------
# Default weights  (override via p_click(..., weights={...}))
# ---------------------------------------------------------------------------
DEFAULT_WEIGHTS: Dict[str, float] = {
    # Logit intercepts (device-specific)
    "beta_0_mobile":      -2.2,
    "beta_0_desktop":     -1.9,
    # Logit feature weights
    "beta_text":           0.6,
    "beta_time":           0.3,
    "beta_whitespace":     0.8,
    # Scroll-decay steepness (below sweet-spot)
    "lam":                 3.0,
}


# ===========================================================================
# Luminance & contrast  (WCAG 2.1 — W3C 2018)
# ===========================================================================

def relative_luminance(r: float, g: float, b: float) -> float:
    """WCAG 2.1 relative luminance.  r, g, b ∈ [0, 255]."""
    def channel(c: float) -> float:
        c_norm = c / 255.0
        if c_norm <= 0.03928:
            return c_norm / 12.92
        return ((c_norm + 0.055) / 1.055) ** 2.4
    return 0.2126 * channel(r) + 0.7152 * channel(g) + 0.0722 * channel(b)


def contrast_ratio(rgb_bg: Tuple[int, int, int],
                   rgb_text: Tuple[int, int, int]) -> float:
    """WCAG 2.1 contrast ratio ∈ [1.0, 21.0].  AA threshold = 4.5."""
    l_bg  = relative_luminance(*rgb_bg)
    l_txt = relative_luminance(*rgb_text)
    light = max(l_bg, l_txt)
    dark  = min(l_bg, l_txt)
    return (light + 0.05) / (dark + 0.05)


def f_contrast(rgb_bg: Tuple[int, int, int],
               rgb_text: Tuple[int, int, int]) -> float:
    """Contrast score ∈ [0, 1], saturates at WCAG AA (4.5:1).
    Returns ~0.22 for identical colours (contrast ratio = 1)."""
    return min(contrast_ratio(rgb_bg, rgb_text) / 4.5, 1.0)


# ===========================================================================
# Colour harmony  (Itten 1961; Ou & Luo, Color Res. Appl. 31(4), 2006)
# ===========================================================================

def _rgb_to_hsv(r: int, g: int, b: int) -> Tuple[float, float, float]:
    """r, g, b ∈ [0, 255]  →  h ∈ [0, 360), s ∈ [0, 1], v ∈ [0, 1]."""
    h, s, v = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
    return h * 360.0, s, v


def _hue_distance(h1: float, h2: float) -> float:
    """Shortest angular distance on the colour wheel ∈ [0, 180]."""
    d = abs(h1 - h2) % 360.0
    return min(d, 360.0 - d)


def f_colour_harmony(rgb_bg: Tuple[int, int, int],
                     rgb_text: Tuple[int, int, int]) -> float:
    """
    Colour harmony score ∈ (0, 1] for (background, text) pair.

    Logic
    -----
    1. If both colours are near-greyscale (saturation < 0.12), harmony is
       irrelevant — only WCAG contrast matters.  Return 0.7 (neutral).

    2. Otherwise score by hue angle gap (Itten 1961 wheel relationships,
       validated by Ou & Luo 2006):

       dH < 15°          same hue → penalty ≈ 0.05–0.10   (text invisible)
       15° ≤ dH < 30°    near-identical → 0.10–0.65         (ramp)
       30° ≤ dH ≤ 60°    analogous → 0.70                   (pleasant, low contrast)
       61° ≤ dH ≤ 99°    mid-range → 0.60                   (neutral)
       100° ≤ dH ≤ 140°  triadic → 0.85
       141° ≤ dH ≤ 165°  split-complementary → 0.90
       166° ≤ dH ≤ 180°  complementary → 1.00               (max contrast + harmony)
    """
    _, s1, _ = _rgb_to_hsv(*rgb_bg)
    _, s2, _ = _rgb_to_hsv(*rgb_text)

    # Both near-greyscale: harmony is undefined; return neutral score
    if s1 < 0.12 and s2 < 0.12:
        return 0.7

    dH = _hue_distance(_rgb_to_hsv(*rgb_bg)[0], _rgb_to_hsv(*rgb_text)[0])

    if dH < 15:
        # Same hue: text and background nearly indistinguishable
        return max(0.05, 0.10 * (dH / 15.0))          # 0.05 → 0.10

    if dH < 30:
        # Near-identical: smooth ramp from same-hue to analogous
        t = (dH - 15.0) / 15.0                         # 0 → 1
        return 0.10 + 0.55 * t                          # 0.10 → 0.65

    if dH <= 60:   return 0.70   # analogous
    if dH <= 99:   return 0.60   # mid-range
    if dH <= 140:  return 0.85   # triadic
    if dH <= 165:  return 0.90   # split-complementary
    return 1.00                  # complementary


# ===========================================================================
# Touch-target size  (Baymard 2024; Apple HIG; Google Material Design)
# ===========================================================================

def f_size(btn_w: float, btn_h: float, device: str) -> float:
    """Touch-target score ∈ [0, 1].  Saturates at 48 px (mobile) / 36 px (desktop)."""
    min_px = 48.0 if device == "mobile" else 36.0
    return min((btn_w * btn_h) / (min_px ** 2), 1.0)


# ===========================================================================
# Vertical position / scroll decay
# (Chartbeat 2014; Nielsen Norman Group / Fessenden 2018)
# ===========================================================================

def f_position_decay(scroll_to_button: float, lam: float = 3.0) -> float:
    """
    p(seen) based on the button's vertical position (scroll_to_button ∈ [0, 1]).

    Sweet-spot model
    ----------------
    scroll = 0     → 0.85  (very top: button appears before value proposition)
    scroll = 0.15  → 1.00  (optimal: just below the hero/value-prop section)
    scroll > 0.15  → exp(−lam × (scroll − 0.15))  exponential decay

    Below-fold decay uses lam = 3.0 (vs old 2.0), calibrated to Chartbeat's
    finding that content below the fold receives roughly 60 % less engagement
    (~0.38 at one fold ≈ 0.33, consistent with exp(−3 × 0.35) ≈ 0.35).

    NNg (2018): primary CTA zone is 10–30 % down the page, hence sweet-spot
    at 0.15 rather than 0.0.
    """
    sweet = 0.15
    if scroll_to_button <= sweet:
        # Linear ramp: 0.85 at scroll=0 → 1.0 at sweet-spot
        return 0.85 + (0.15 / sweet) * scroll_to_button
    return float(np.exp(-lam * (scroll_to_button - sweet)))



# ===========================================================================
# Time of day  (Infolinks, 1 T impressions)
# ===========================================================================

def f_time(hour: int) -> float:
    """Time-of-day multiplier: peak 1.14, night 0.90, otherwise 1.0."""
    if 11 <= hour <= 14:
        return 1.14
    if hour >= 23 or hour <= 5:
        return 0.90
    return 1.0


# ===========================================================================
# Whitespace  (VWO / Open Mile study)
# ===========================================================================

def f_whitespace(ratio: float) -> float:
    """Sigmoid transition centred at 0.2 (VWO study)."""
    return 1.0 / (1.0 + np.exp(-10.0 * (ratio - 0.2)))


# ===========================================================================
# Penalties
# ===========================================================================

def penalty_overflow(font_size: float, btn_h: float) -> float:
    """
    Returns 1 if font fits inside the button (font ≤ 60 % of btn_h).
    Decays as max_fit / font_size when text overflows — goes to 0 as
    font_size → ∞.  Ensures overflow configs cannot score near-optimally.
    """
    if btn_h <= 0:
        return 0.0
    max_fit = 0.6 * btn_h
    if font_size <= max_fit:
        return 1.0
    return max_fit / font_size


def f_margin_balance(btn_w: float, btn_h: float,
                     font_size: float, whitespace_ratio: float) -> float:
    """
    Penalises uneven horizontal vs vertical text clearance inside the button.

    Derivation
    ----------
    horizontal clearance per side ≈ whitespace_ratio × btn_w   (px)
    vertical clearance per side   = max(btn_h − font_size, 0) / 2   (px)

    score = 0.5 + 0.5 × (smaller / larger)
            → 1.0 when perfectly balanced
            → 0.5 when one clearance is tiny compared to the other
            → 0.3 when exactly one clearance is zero
            → 0.0 when both are zero (fully packed)

    References
    ----------
    Lidwell, Holden & Butler, "Universal Principles of Design" (2010) —
    Symmetry / White Space principle.
    Google Material Design 2024 — "Spacing within components should be
    consistent horizontally and vertically."
    """
    if btn_w <= 0 or btn_h <= 0:
        return 0.0

    h_cl = whitespace_ratio * btn_w / 2.0            # horizontal clearance (px)
    v_cl = max(btn_h - font_size, 0.0) / 2.0         # vertical clearance (px)

    if h_cl <= 0 and v_cl <= 0:
        return 0.0
    if h_cl <= 0 or v_cl <= 0:
        return 0.3   # one side has zero clearance

    ratio = min(h_cl, v_cl) / max(h_cl, v_cl)
    return 0.5 + 0.5 * ratio



# ===========================================================================
# Main entry point
# ===========================================================================

def p_click(params: Dict, weights: Optional[Dict[str, float]] = None) -> float:
    """
    Compute p(click) for one button configuration.

    Required keys in params
    -----------------------
    rgb_bg, rgb_text      : (r, g, b)  each channel ∈ [0, 255]
    btn_w, btn_h          : button dimensions in px
    font_size             : font size in px
    text_quality          : ∈ [0, 1]
    whitespace_ratio      : ∈ [0, 0.5]  padding / button-width ratio
    scroll_to_button      : ∈ [0, 1]  normalised vertical position on page
    hour                  : int ∈ [0, 23]
    device                : 'mobile' | 'desktop'

    Optional weights dict overrides any key in DEFAULT_WEIGHTS.

    Return
    ------
    float ∈ [0, 1]
    """
    w = DEFAULT_WEIGHTS if weights is None else {**DEFAULT_WEIGHTS, **weights}
    device = params["device"]

    # Device-specific intercept and size weight
    beta_0    = w["beta_0_mobile"]    if device == "mobile" else w["beta_0_desktop"]

    # ---- signal scores (all in [0, 1]) ----
    contrast_score = f_contrast(params["rgb_bg"], params["rgb_text"])
    size_score     = f_size(params["btn_w"], params["btn_h"], device)
    ws_score       = f_whitespace(params["whitespace_ratio"])
    time_score     = f_time(params["hour"])

    # ---- logit (attractiveness) ----
    # Colour harmony is NOT in the logit — it enters only as a multiplicative
    # penalty.  This keeps the logit's semantics as a pure accessibility /
    # usability signal and avoids double-counting harmony.
    logit = (
        beta_0         
        + w["beta_text"]       * float(params["text_quality"])
        + w["beta_time"]       * (time_score - 1.0)
        + w["beta_whitespace"] * ws_score
    )
    p_attractive = 1.0 / (1.0 + np.exp(-logit))

    # ---- multiplicative penalties ----
    # Each factor ∈ (0, 1].  Any single factor near 0 collapses p_click.
    penalty = (
        penalty_overflow(params["font_size"], params["btn_h"])   # text fits in button
        * contrast_score                                          # WCAG readability
        * size_score                                             # touch target
        * f_colour_harmony(params["rgb_bg"], params["rgb_text"]) # colour wheel
        * f_margin_balance(params["btn_w"], params["btn_h"],
                           params["font_size"],
                           float(params["whitespace_ratio"]))    # even padding
    )

    # ---- visibility (position on page) ----
    visibility = f_position_decay(
        float(params["scroll_to_button"]), lam=w["lam"]
    )

    return float(np.clip(p_attractive * penalty * visibility, 0.0, 1.0))


