"""
=============================================================
  blackbox_benchmarks — single-file distribution
  A/B Testing as a Black-Box Experiment + Multi-Domain Suite
  Author: Sapunov Artemiy Maximovich, HSE БПАД244
=============================================================

PACKAGE STRUCTURE (when used as a package, split into modules):
  blackbox_benchmarks/
    __init__.py
    benchmarks/
      base.py          <- BlackBoxBenchmark (ABC)
      mathematics.py   <- Rosenbrock, Rastrigin, Ackley
      physics.py       <- LennardJones, HarmonicOscillator, DoublePotentialWell
      chemistry.py     <- MuellerBrown, ArrheniusRate
      engineering.py   <- TensionCompressionSpring, WeldedBeam, PressureVessel
      ab_testing.py    <- ABTestingBenchmark
      economics.py     <- CobbDouglas, UtilityMaximisation
    utils/
      visualization.py <- plot_1d, plot_2d_contour, plot_ab_test_results

QUICK START:
  from blackbox_benchmarks_full import Rosenbrock, ABTestingBenchmark
  b = Rosenbrock()
  print(b([1.0, 1.0]))          # 0.0  — black-box call
  print(b.gradient([0.5, 0.2])) # analytical gradient via Sympy
  print(b.optimum())            # known global minimum

DEPENDENCIES:
  pip install sympy numpy matplotlib scipy
"""

# ================================================================
# Imports
# ================================================================

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Union


# ================================================================
# BASE CLASS
# ================================================================

class BlackBoxBenchmark(ABC):
    """
    Abstract base class for all black-box benchmark functions.

    The optimizer sees only __call__. Sympy internals (expr, gradient,
    Hessian) are available for analysis/documentation purposes.
    """

    name: str = "AbstractBenchmark"
    domain: str = "abstract"

    def __init__(self):
        self.vars: List[sp.Symbol] = self._define_vars()
        self.expr: sp.Expr = self._define_expr()
        self._f_numeric = sp.lambdify(self.vars, self.expr, modules="numpy")

    @abstractmethod
    def _define_vars(self) -> List[sp.Symbol]:
        """Return list of sympy symbols."""

    @abstractmethod
    def _define_expr(self) -> sp.Expr:
        """Return the sympy expression."""

    @abstractmethod
    def bounds(self) -> List[Tuple[float, float]]:
        """Return list of (lower, upper) bounds per variable."""

    @abstractmethod
    def optimum(self) -> Tuple[Optional[np.ndarray], Optional[float]]:
        """Return (x_opt, f_opt). Return (None, None) if stochastic."""

    def __call__(self, x: Union[List[float], np.ndarray]) -> float:
        """Black-box evaluation at x. The only optimizer-facing method."""
        x = np.asarray(x, dtype=float)
        if x.shape != (len(self.vars),):
            raise ValueError(
                f"{self.name} expects shape ({len(self.vars)},), got {x.shape}."
            )
        return float(self._f_numeric(*x))

    def gradient_expr(self) -> List[sp.Expr]:
        """Symbolic gradient."""
        return [sp.diff(self.expr, v) for v in self.vars]

    def gradient(self, x: Union[List[float], np.ndarray]) -> np.ndarray:
        """Evaluate analytical gradient at x."""
        x = np.asarray(x, dtype=float)
        funcs = [sp.lambdify(self.vars, g, modules="numpy") for g in self.gradient_expr()]
        return np.array([float(f(*x)) for f in funcs])

    def hessian_expr(self) -> List[List[sp.Expr]]:
        """Symbolic Hessian."""
        n = len(self.vars)
        return [[sp.diff(self.expr, self.vars[i], self.vars[j]) for j in range(n)]
                for i in range(n)]

    def hessian(self, x: Union[List[float], np.ndarray]) -> np.ndarray:
        """Evaluate analytical Hessian at x."""
        x = np.asarray(x, dtype=float)
        n = len(self.vars)
        H = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                f_ij = sp.lambdify(self.vars, self.hessian_expr()[i][j], modules="numpy")
                H[i, j] = float(f_ij(*x))
        return H

    def ndim(self) -> int:
        return len(self.vars)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(domain='{self.domain}', ndim={self.ndim()})"


# ================================================================
# DOMAIN: MATHEMATICS
# ================================================================

class Rosenbrock(BlackBoxBenchmark):
    """
    Rosenbrock (banana) function.
    f(x,y) = (a-x)^2 + b*(y-x^2)^2
    Global minimum: f(a, a^2) = 0. Default a=1, b=100.
    """
    name = "Rosenbrock"; domain = "mathematics"

    def __init__(self, a: float = 1.0, b: float = 100.0):
        self.a = a; self.b = b; super().__init__()

    def _define_vars(self): return list(sp.symbols("x y"))
    def _define_expr(self):
        x, y = self.vars
        return (self.a - x)**2 + self.b*(y - x**2)**2
    def bounds(self): return [(-2.0, 2.0), (-1.0, 3.0)]
    def optimum(self): return np.array([self.a, self.a**2]), 0.0


class Rastrigin(BlackBoxBenchmark):
    """
    Rastrigin function — highly multimodal.
    f(x,y) = 2A + x^2 - A*cos(2pi*x) + y^2 - A*cos(2pi*y)
    Global minimum: f(0,0) = 0. Default A=10.
    """
    name = "Rastrigin"; domain = "mathematics"

    def __init__(self, A: float = 10.0):
        self.A = A; super().__init__()

    def _define_vars(self): return list(sp.symbols("x y"))
    def _define_expr(self):
        x, y = self.vars; A = self.A
        return 2*A + (x**2 - A*sp.cos(2*sp.pi*x)) + (y**2 - A*sp.cos(2*sp.pi*y))
    def bounds(self): return [(-5.12, 5.12), (-5.12, 5.12)]
    def optimum(self): return np.array([0.0, 0.0]), 0.0


class Ackley(BlackBoxBenchmark):
    """
    Ackley function. Global minimum: f(0,0) = 0.
    """
    name = "Ackley"; domain = "mathematics"

    def _define_vars(self): return list(sp.symbols("x y"))
    def _define_expr(self):
        x, y = self.vars
        return (-20*sp.exp(-0.2*sp.sqrt(0.5*(x**2+y**2)))
                - sp.exp(0.5*(sp.cos(2*sp.pi*x)+sp.cos(2*sp.pi*y)))
                + sp.E + 20)
    def bounds(self): return [(-5.0, 5.0), (-5.0, 5.0)]
    def optimum(self): return np.array([0.0, 0.0]), 0.0


# ================================================================
# DOMAIN: PHYSICS
# ================================================================

class LennardJones(BlackBoxBenchmark):
    """
    Lennard-Jones pair potential.
    V(r) = 4*eps*[(sig/r)^12 - (sig/r)^6]
    Global minimum: V(2^(1/6)*sig) = -eps. Default eps=sig=1.
    """
    name = "LennardJones"; domain = "physics"

    def __init__(self, epsilon: float = 1.0, sigma: float = 1.0):
        self.epsilon = epsilon; self.sigma = sigma; super().__init__()

    def _define_vars(self): return [sp.Symbol("r", positive=True)]
    def _define_expr(self):
        r = self.vars[0]; eps, sig = self.epsilon, self.sigma
        return 4*eps*((sig/r)**12 - (sig/r)**6)
    def bounds(self): return [(0.8*self.sigma, 3.0*self.sigma)]
    def optimum(self):
        return np.array([2**(1/6)*self.sigma]), -self.epsilon


class HarmonicOscillator(BlackBoxBenchmark):
    """
    2D coupled harmonic oscillator potential.
    V(x,y) = 0.5*k1*x^2 + 0.5*k2*y^2 + c*x*y
    Global minimum: V(0,0) = 0. Default k1=1, k2=2, c=0.5.
    """
    name = "HarmonicOscillator"; domain = "physics"

    def __init__(self, k1=1.0, k2=2.0, c=0.5):
        self.k1=k1; self.k2=k2; self.c=c; super().__init__()

    def _define_vars(self): return list(sp.symbols("x y"))
    def _define_expr(self):
        x, y = self.vars
        return sp.Rational(1,2)*self.k1*x**2 + sp.Rational(1,2)*self.k2*y**2 + self.c*x*y
    def bounds(self): return [(-5.0, 5.0), (-5.0, 5.0)]
    def optimum(self): return np.array([0.0, 0.0]), 0.0


class DoublePotentialWell(BlackBoxBenchmark):
    """
    1D double-well potential: V(x) = a*x^4 - b*x^2 + c*x
    Default a=1, b=4, c=0 (symmetric wells).
    """
    name = "DoublePotentialWell"; domain = "physics"

    def __init__(self, a=1.0, b=4.0, c=0.0):
        self.a=a; self.b=b; self.c=c; super().__init__()

    def _define_vars(self): return [sp.Symbol("x")]
    def _define_expr(self):
        x = self.vars[0]
        return self.a*x**4 - self.b*x**2 + self.c*x
    def bounds(self): return [(-3.0, 3.0)]
    def optimum(self):
        import math
        x_opt = math.sqrt(self.b / (2*self.a))
        candidates = [np.array([x_opt]), np.array([-x_opt])]
        best = min(candidates, key=lambda xc: self(xc))
        return best, float(self(best))


# ================================================================
# DOMAIN: CHEMISTRY
# ================================================================

class MuellerBrown(BlackBoxBenchmark):
    """
    Mueller-Brown potential energy surface (2D, 3 minima).
    Reference: Mueller & Brown, Theor. Chim. Acta 53 (1979).
    """
    name = "MuellerBrown"; domain = "chemistry"
    _A=[-200,-100,-170,15]; _a=[-1,-1,-6.5,0.7]; _b=[0,0,11,0.6]
    _c=[-10,-10,-6.5,0.7]; _x0=[1,0,-0.5,-1]; _y0=[0,0.5,1.5,1]

    def _define_vars(self): return list(sp.symbols("x y"))
    def _define_expr(self):
        x, y = self.vars; total = sp.Integer(0)
        for k in range(4):
            exp = (self._a[k]*(x-self._x0[k])**2
                   + self._b[k]*(x-self._x0[k])*(y-self._y0[k])
                   + self._c[k]*(y-self._y0[k])**2)
            total += self._A[k]*sp.exp(exp)
        return total
    def bounds(self): return [(-1.5, 1.2), (-0.2, 2.0)]
    def optimum(self):
        x_opt = np.array([-0.558, 1.442])
        return x_opt, float(self(x_opt))


class ArrheniusRate(BlackBoxBenchmark):
    """
    Negative Arrhenius rate (minimise = maximise rate).
    f(T, Ea) = -A * exp(-Ea / (R*T))
    Default A=1e13, R=8.314e-3 kJ/(mol·K).
    """
    name = "ArrheniusRate"; domain = "chemistry"

    def __init__(self, A_pre=1e13, R=8.314e-3):
        self.A_pre=A_pre; self.R=R; super().__init__()

    def _define_vars(self): return list(sp.symbols("T Ea", positive=True))
    def _define_expr(self):
        T, Ea = self.vars
        return -self.A_pre*sp.exp(-Ea/(self.R*T))
    def bounds(self): return [(200.0, 1000.0), (10.0, 200.0)]
    def optimum(self):
        x_opt = np.array([1000.0, 10.0])
        return x_opt, float(self(x_opt))


# ================================================================
# DOMAIN: ENGINEERING
# ================================================================

class TensionCompressionSpring(BlackBoxBenchmark):
    """Spring weight: f(d,D,N) = (N+2)*D*d^2"""
    name = "TensionCompressionSpring"; domain = "engineering"
    def _define_vars(self): return list(sp.symbols("d D N", positive=True))
    def _define_expr(self):
        d, D, N = self.vars; return (N+2)*D*d**2
    def bounds(self): return [(0.05,2.0),(0.25,1.3),(2.0,15.0)]
    def optimum(self):
        x = np.array([0.052,0.358,11.5]); return x, float(self(x))


class WeldedBeam(BlackBoxBenchmark):
    """Welded beam cost: f(h,l,t,b) = 1.10471*h^2*l + 0.04811*t*b*(14+l)"""
    name = "WeldedBeam"; domain = "engineering"
    def _define_vars(self): return list(sp.symbols("h l t b", positive=True))
    def _define_expr(self):
        h,l,t,b = self.vars
        return 1.10471*h**2*l + 0.04811*t*b*(14+l)
    def bounds(self): return [(0.1,2.0),(0.1,10.0),(0.1,10.0),(0.1,2.0)]
    def optimum(self):
        x = np.array([0.2057,3.4705,9.0366,0.2057]); return x, float(self(x))


class PressureVessel(BlackBoxBenchmark):
    """Pressure vessel cost minimisation (4 variables)."""
    name = "PressureVessel"; domain = "engineering"
    def _define_vars(self): return list(sp.symbols("Ts Th R L", positive=True))
    def _define_expr(self):
        Ts,Th,R,L = self.vars
        return 0.6224*Ts*R*L + 1.7781*Th*R**2 + 3.1661*Ts**2*L + 19.84*Ts**2*R
    def bounds(self): return [(0.0625,6.1875),(0.0625,6.1875),(10.0,200.0),(10.0,200.0)]
    def optimum(self):
        x = np.array([0.8125,0.4375,42.098,176.637]); return x, float(self(x))


# ================================================================
# DOMAIN: A/B TESTING
# ================================================================

class ABTestingBenchmark(BlackBoxBenchmark):
    """
    Simulated A/B test over two recommender variants.

    The optimizer controls traffic split theta = [theta_A, theta_B],
    theta_A + theta_B = 1. The black box simulates n_users Bernoulli
    trials and returns negative weighted revenue (minimise = maximise).

    The Sympy expression is an analytical surrogate for gradient utilities.
    The actual __call__ is stochastic (seeded RNG).

    Parameters
    ----------
    cvr_A : float   True conversion rate of variant A (control).
    cvr_B : float   True conversion rate of variant B (treatment).
    revenue_per_conversion : float
    n_users : int
    seed : int
    """
    name = "ABTestingBenchmark"; domain = "ab_testing"

    def __init__(self, cvr_A=0.027, cvr_B=0.042,
                 revenue_per_conversion=10.0, n_users=10_000, seed=42):
        self.cvr_A = cvr_A; self.cvr_B = cvr_B
        self.revenue_per_conversion = revenue_per_conversion
        self.n_users = n_users
        self.rng = np.random.default_rng(seed)
        super().__init__()

    def _define_vars(self): return list(sp.symbols("theta_A theta_B"))
    def _define_expr(self):
        tA, tB = self.vars; rev = self.revenue_per_conversion
        return -(tA*self.cvr_A + tB*self.cvr_B)*rev

    def __call__(self, x: np.ndarray) -> float:
        """Stochastic black-box: simulate the A/B experiment."""
        x = np.asarray(x, dtype=float)
        if x.shape != (2,):
            raise ValueError(f"Expected shape (2,), got {x.shape}.")
        theta_A, theta_B = x
        if not np.isclose(theta_A + theta_B, 1.0, atol=1e-6):
            raise ValueError("theta_A + theta_B must equal 1.")
        n_A = int(round(theta_A * self.n_users))
        n_B = self.n_users - n_A
        conv_A = self.rng.binomial(n_A, self.cvr_A) if n_A > 0 else 0
        conv_B = self.rng.binomial(n_B, self.cvr_B) if n_B > 0 else 0
        return -(conv_A + conv_B) * self.revenue_per_conversion

    def bounds(self): return [(0.0, 1.0), (0.0, 1.0)]
    def optimum(self):
        x_opt = np.array([0.0, 1.0]) if self.cvr_B >= self.cvr_A else np.array([1.0, 0.0])
        return x_opt, float(self._f_numeric(*x_opt))

    def run_statistical_test(self, theta_A: float = 0.5) -> dict:
        """
        Run a two-proportion z-test at the given traffic split.
        Returns a dict with z_stat, p_value, 95% CI, SRM check, etc.
        """
        from scipy import stats
        theta_B = 1.0 - theta_A
        n_A = int(round(theta_A * self.n_users)); n_B = self.n_users - n_A
        conv_A = self.rng.binomial(n_A, self.cvr_A)
        conv_B = self.rng.binomial(n_B, self.cvr_B)
        cvr_hat_A = conv_A / n_A; cvr_hat_B = conv_B / n_B
        delta = cvr_hat_B - cvr_hat_A
        se = np.sqrt(cvr_hat_A*(1-cvr_hat_A)/n_A + cvr_hat_B*(1-cvr_hat_B)/n_B)
        z = delta / se if se > 0 else 0.0
        p_value = 2*(1 - stats.norm.cdf(abs(z)))
        n_total = n_A + n_B
        srm_chi2 = ((n_A - n_total/2)**2 + (n_B - n_total/2)**2) / (n_total/2)
        srm_p = 1 - stats.chi2.cdf(srm_chi2, df=1)
        return {
            "n_A": n_A, "n_B": n_B,
            "cvr_hat_A": round(cvr_hat_A, 6), "cvr_hat_B": round(cvr_hat_B, 6),
            "delta": round(delta, 6), "se": round(se, 6),
            "z_stat": round(z, 4), "p_value": round(p_value, 6),
            "ci_low": round(delta - 1.96*se, 6),
            "ci_high": round(delta + 1.96*se, 6),
            "srm_chi2": round(srm_chi2, 4), "srm_p": round(srm_p, 4),
            "significant": p_value < 0.05,
        }


# ================================================================
# DOMAIN: ECONOMICS
# ================================================================

class CobbDouglas(BlackBoxBenchmark):
    """
    Negative Cobb-Douglas production function.
    f(K,L) = -A * K^alpha * L^beta
    Default A=1, alpha=0.3, beta=0.7.
    """
    name = "CobbDouglas"; domain = "economics"

    def __init__(self, A=1.0, alpha=0.3, beta=0.7):
        self.A_param=A; self.alpha=alpha; self.beta=beta; super().__init__()

    def _define_vars(self): return list(sp.symbols("K L", positive=True))
    def _define_expr(self):
        K, L = self.vars
        return -self.A_param * K**self.alpha * L**self.beta
    def bounds(self): return [(0.1, 100.0), (0.1, 100.0)]
    def optimum(self):
        x = np.array([100.0, 100.0]); return x, float(self(x))


class UtilityMaximisation(BlackBoxBenchmark):
    """
    Negative CES utility: U(x1,x2) = -(x1^rho + x2^rho)^(1/rho)
    Default rho=0.5.
    """
    name = "UtilityMaximisation"; domain = "economics"

    def __init__(self, rho=0.5):
        self.rho=rho; super().__init__()

    def _define_vars(self): return list(sp.symbols("x1 x2", positive=True))
    def _define_expr(self):
        x1, x2 = self.vars
        return -(x1**self.rho + x2**self.rho)**(1/self.rho)
    def bounds(self): return [(0.1, 10.0), (0.1, 10.0)]
    def optimum(self):
        x = np.array([10.0, 10.0]); return x, float(self(x))


# ================================================================
# REGISTRY
# ================================================================

REGISTRY = {
    "mathematics": [Rosenbrock, Rastrigin, Ackley],
    "physics":     [LennardJones, HarmonicOscillator, DoublePotentialWell],
    "chemistry":   [MuellerBrown, ArrheniusRate],
    "engineering": [TensionCompressionSpring, WeldedBeam, PressureVessel],
    "ab_testing":  [ABTestingBenchmark],
    "economics":   [CobbDouglas, UtilityMaximisation],
}
ALL_BENCHMARKS = [cls for classes in REGISTRY.values() for cls in classes]


def list_benchmarks(domain: str = None) -> None:
    """Print all available benchmarks, optionally filtered by domain."""
    for dom, classes in REGISTRY.items():
        if domain and dom != domain:
            continue
        print(f"\n[{dom}]")
        for cls in classes:
            try:
                b = cls()
                x_opt, f_opt = b.optimum()
                opt_str = f"f*={f_opt:.4g}" if f_opt is not None else "stochastic"
                print(f"  {cls.__name__:<35} ndim={b.ndim()}  {opt_str}")
            except Exception as e:
                print(f"  {cls.__name__:<35} (error: {e})")


# ================================================================
# VISUALIZATION UTILITIES
# ================================================================

def plot_1d(benchmark, n_points=500, show_optimum=True, ax=None, title=None):
    """Plot a 1D benchmark function over its bounds."""
    if benchmark.ndim() != 1:
        raise ValueError(f"plot_1d requires ndim=1, got {benchmark.ndim()}")
    lo, hi = benchmark.bounds()[0]
    xs = np.linspace(lo, hi, n_points)
    ys = np.array([benchmark(np.array([x])) for x in xs])
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(xs, ys, lw=2, color="steelblue", label=benchmark.name)
    ax.set_xlabel(str(benchmark.vars[0])); ax.set_ylabel("f(x)")
    ax.set_title(title or benchmark.name)
    if show_optimum:
        x_opt, f_opt = benchmark.optimum()
        if x_opt is not None:
            ax.axvline(x_opt[0], color="crimson", ls="--", alpha=0.7,
                       label=f"Optimum x={x_opt[0]:.3f}")
            ax.scatter([x_opt[0]], [f_opt], color="crimson", zorder=5)
    ax.legend(); ax.grid(True, alpha=0.3)
    if fig:
        fig.tight_layout()
    return fig or ax.get_figure()


def plot_2d_contour(benchmark, n_grid=200, show_optimum=True,
                    show_gradient=False, gradient_density=15,
                    log_scale=False, ax=None, title=None):
    """Filled contour plot for a 2D benchmark function."""
    if benchmark.ndim() != 2:
        raise ValueError(f"plot_2d_contour requires ndim=2, got {benchmark.ndim()}")
    (x_lo, x_hi), (y_lo, y_hi) = benchmark.bounds()
    xs = np.linspace(x_lo, x_hi, n_grid); ys = np.linspace(y_lo, y_hi, n_grid)
    X, Y = np.meshgrid(xs, ys)
    Z = np.array([[benchmark(np.array([X[i,j], Y[i,j]])) for j in range(n_grid)]
                  for i in range(n_grid)])
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))
    norm = mcolors.LogNorm(vmin=max(Z.min(), 1e-6), vmax=Z.max()) if log_scale else None
    cf = ax.contourf(X, Y, Z, levels=50, cmap="viridis", norm=norm, alpha=0.85)
    ax.contour(X, Y, Z, levels=20, colors="white", linewidths=0.4, alpha=0.5)
    plt.colorbar(cf, ax=ax, label="f(x, y)")
    if show_gradient:
        xg = np.linspace(x_lo, x_hi, gradient_density)
        yg = np.linspace(y_lo, y_hi, gradient_density)
        Xg, Yg = np.meshgrid(xg, yg)
        U, V = np.zeros_like(Xg), np.zeros_like(Yg)
        for i in range(gradient_density):
            for j in range(gradient_density):
                try:
                    g = benchmark.gradient(np.array([Xg[i,j], Yg[i,j]]))
                    n = np.linalg.norm(g) + 1e-10
                    U[i,j] = -g[0]/n; V[i,j] = -g[1]/n
                except Exception:
                    pass
        ax.quiver(Xg, Yg, U, V, color="white", alpha=0.6, scale=gradient_density*1.5)
    if show_optimum:
        x_opt, _ = benchmark.optimum()
        if x_opt is not None:
            ax.scatter(x_opt[0], x_opt[1], color="red", s=120, zorder=10, marker="*",
                       label=f"Optimum ({x_opt[0]:.2f}, {x_opt[1]:.2f})")
            ax.legend(loc="upper right", fontsize=9)
    ax.set_xlabel(str(benchmark.vars[0])); ax.set_ylabel(str(benchmark.vars[1]))
    ax.set_title(title or benchmark.name)
    if fig:
        fig.tight_layout()
    return fig or ax.get_figure()


def plot_ab_test_results(results: dict, title: str = "A/B Test Results"):
    """Visualise A/B statistical test results."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(title, fontsize=13, fontweight="bold")
    ax = axes[0]
    variants = ["Variant A\n(Control)", "Variant B\n(Treatment)"]
    cvrs = [results["cvr_hat_A"], results["cvr_hat_B"]]
    colors = ["#4C72B0", "#DD8452"]
    bars = ax.bar(variants, cvrs, color=colors, width=0.5, edgecolor="black")
    for bar, cvr in zip(bars, cvrs):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.0005,
                f"{cvr:.4f}", ha="center", va="bottom", fontsize=11)
    se = results["se"]; delta = results["delta"]
    ax.errorbar([1], [cvrs[1]],
                yerr=[[abs(delta-results["ci_low"])],[abs(results["ci_high"]-delta)]],
                fmt="none", color="black", capsize=6)
    sig_text = "✓ Significant" if results["significant"] else "✗ Not Significant"
    ax.set_title(f"Conversion Rates  |  {sig_text}",
                 color="green" if results["significant"] else "red")
    ax.set_ylabel("Estimated CVR"); ax.set_ylim(0, max(cvrs)*1.3); ax.grid(axis="y", alpha=0.3)
    ax2 = axes[1]
    null_deltas = np.random.normal(0, se, 10000)
    ax2.hist(null_deltas, bins=60, density=True, color="lightsteelblue",
             edgecolor="white", alpha=0.8, label="H₀ distribution")
    ax2.axvline(delta, color="crimson", lw=2, label=f"Observed Δ={delta:.4f}")
    ax2.axvline(results["ci_low"], color="orange", lw=1.5, ls=":", label="95% CI")
    ax2.axvline(results["ci_high"], color="orange", lw=1.5, ls=":")
    ax2.set_title(f"Z={results['z_stat']:.3f}   p={results['p_value']:.4f}")
    ax2.set_xlabel("Δ CVR"); ax2.set_ylabel("Density"); ax2.legend(fontsize=9); ax2.grid(alpha=0.3)
    fig.tight_layout()
    return fig


# ================================================================
# DEMO / ENTRY POINT
# ================================================================

if __name__ == "__main__":
    print("blackbox_benchmarks v0.1.0\n")
    list_benchmarks()

    print("\n--- Rosenbrock ---")
    rb = Rosenbrock()
    print(f"  expr     : {rb.expr}")
    print(f"  f(1,1)   : {rb([1.0, 1.0]):.6f}  (expected 0.0)")
    print(f"  grad(0,0): {rb.gradient([0.0, 0.0])}")

    print("\n--- LennardJones ---")
    lj = LennardJones()
    r_min = 2**(1/6)
    print(f"  V(r_min) : {lj([r_min]):.6f}  (expected -1.0)")

    print("\n--- ABTestingBenchmark ---")
    ab = ABTestingBenchmark()
    print(f"  surrogate: {ab.expr}")
    print(f"  f([0,1]) : {ab([0.0, 1.0]):.2f}  (all traffic to B)")
    results = ab.run_statistical_test(theta_A=0.5)
    print(f"  z={results['z_stat']}  p={results['p_value']}  sig={results['significant']}")

    print("\nGenerating plots ...")
    fig1 = plot_2d_contour(Rosenbrock(), log_scale=True, show_gradient=True,
                           title="Rosenbrock (log scale + gradient)")
    fig2 = plot_2d_contour(Rastrigin(), title="Rastrigin")
    fig3 = plot_1d(LennardJones(), title="Lennard-Jones (Physics)")
    fig4 = plot_ab_test_results(results)
    plt.show()
    print("Done.")