import os
import contextlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.style as mstyle
from typing import Optional, List, Tuple, Union, Any, Dict
from datetime import datetime
from agrilyzer.analyzers import Agrilyzer
from agrilyzer.config import cfg


class Visualizer:
    """High-level plotting util around **Agrilyzer** with optional themes.

    Uses built-in matplotlib and seaborn styles only.
    """

    def __init__(
        self,
        agrilyzer: Agrilyzer,
        *,
        font_scale: Optional[float] = None,
        user_theme: Union[bool, str, None] = None,
        user_cmap: str = "viridis",
    ) -> None:
        self.client = agrilyzer
        self.name = agrilyzer.name
        self.PARAMS = agrilyzer.PARAMS
        self.user_theme = user_theme
        self.user_cmap = user_cmap

        # plotting configuration
        self.plot_cfg: Dict[str, Any] = cfg.get("plot_config", {})
        self.font_scale = font_scale

        # figure & line defaults
        self.fig_defaults: Dict[str, Any] = self.plot_cfg.get("figure", {})
        self.line_defaults: Dict[str, Any] = self.plot_cfg.get("line", {})

    def _style_context(self, override: Union[bool, str, None] = None):
        """Return a context manager applying built-in style theme if requested."""
        choice = override if override is not None else self.user_theme
        if not choice:
            return contextlib.nullcontext()

        # Determine style name
        if choice is True:
            style_name = self.plot_cfg.get("theme", "ggplot")
        elif isinstance(choice, str):
            style_name = choice
        else:
            return contextlib.nullcontext()

        # Apply style if available, else fallback
        if style_name in mstyle.available:
            return plt.style.context(style_name)
        fallback = self.plot_cfg.get("theme", "ggplot")
        return plt.style.context(fallback)

    def _apply_font_scaling(self, fig: plt.Figure, base_size: float = 12) -> None:
        w, h = fig.get_size_inches()
        factor = self.font_scale if self.font_scale else min(w, h) / 6
        size = base_size * factor
        plt.rcParams.update(
            {
                "axes.titlesize": size * 1.1,
                "axes.labelsize": size,
                "xtick.labelsize": size * 0.8,
                "ytick.labelsize": size * 0.8,
                "legend.fontsize": size * 0.9,
            }
        )

    def plot(
        self,
        cols: Optional[List[str]] = None,
        *,
        ma: int = 0,
        dual: bool = False,
        figsize: Tuple[int, int] = (12, 6),
        title: Optional[str] = None,
        save_path: Optional[str] = None,
        date_range: Optional[Tuple[Union[str, datetime], Union[str, datetime]]] = None,
        fig_kwargs: Optional[Dict[str, Any]] = None,
        line_kwargs: Optional[Dict[str, Any]] = None,
        apply_theme: Union[bool, str, None] = None,
    ) -> None:
        df = self.client.load_df()
        if date_range:
            s, e = map(pd.to_datetime, date_range)
            df = df[df.date.between(s, e)]

        cols = cols or [c for c in df.columns if c != "date"]
        missing = set(cols) - set(df.columns)
        if missing:
            raise KeyError(f"Columns not found: {missing}")

        with self._style_context(apply_theme):
            fig_kw = {**self.fig_defaults, **(fig_kwargs or {})}
            fig, ax = plt.subplots(figsize=figsize, **fig_kw)
            self._apply_font_scaling(fig)
            ax2 = ax.twinx() if dual and len(cols) > 1 else None
            cmap = plt.cm.get_cmap(self.user_cmap, len(cols))

            for i, code in enumerate(cols):
                srs = df[["date", code]].copy()
                if ma:
                    srs[code] = srs[code].rolling(ma).mean()
                label = self.PARAMS.get(code, code)
                tgt_ax = ax2 if dual and i == 1 else ax

                ln_kw = {**self.line_defaults, **(line_kwargs or {})}
                ln_kw.setdefault("color", cmap(i))
                ln_kw.setdefault("label", label)
                tgt_ax.plot(srs.date, srs[code], **ln_kw)

            ax.set_xlabel("Date")
            ax.set_ylabel(
                "Value" if len(cols) > 1 else self.PARAMS.get(cols[0], cols[0])
            )
            ax.set_title(
                title
                or f"{self.name}: {self.client.start.date()} → {self.client.end.date()}"
            )
            ax.grid(True)
            ax.legend()
            fig.tight_layout()

            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                fig.savefig(save_path, dpi=300)
            plt.show()

    def overlay(
        self,
        other: pd.DataFrame,
        other_cols: List[str],
        *,
        weather_cols: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (12, 6),
        title: Optional[str] = None,
        save_path: Optional[str] = None,
        date_range: Optional[Tuple[Union[str, datetime], Union[str, datetime]]] = None,
        agri_ma: int = 0,
        weather_ma: int = 0,
        agri_kind: str = "errorbar",
        weather_kind: str = "line",
        fig_kwargs: Optional[Dict[str, Any]] = None,
        agri_plot_kwargs: Optional[Dict[str, Any]] = None,
        weather_plot_kwargs: Optional[Dict[str, Any]] = None,
        apply_theme: Union[bool, str, None] = None,
    ) -> Tuple[plt.Figure, plt.Axes, plt.Axes]:
        with self._style_context(apply_theme):
            df_ag = other.copy()
            df_ag["date"] = pd.to_datetime(df_ag["date"])
            df_ag.set_index("date", inplace=True)
            if agri_ma > 1:
                df_ag = (
                    df_ag[other_cols]
                    .rolling(agri_ma, center=True)
                    .mean()
                    .dropna()
                    .reset_index()
                )
            else:
                df_ag = df_ag.reset_index()
            stats = self._prepare_agri_stats(df_ag, other_cols)

            df_w = self._prepare_weather(weather_cols)
            if weather_ma > 1:
                df_w = df_w.rolling(weather_ma, center=True).mean().dropna()

            if date_range:
                start, end = map(pd.to_datetime, date_range)
                stats = stats.loc[start:end]
                df_w = df_w.loc[start:end]

            fig_kw = {**self.fig_defaults, **(fig_kwargs or {})}
            fig, ax1 = plt.subplots(figsize=figsize, **fig_kw)
            self._apply_font_scaling(fig)
            ax2 = ax1.twinx()

            ag_h, ag_l = self._plot_agri(
                ax1,
                stats,
                other_cols,
                kind=agri_kind,
                plot_kwargs=agri_plot_kwargs or {},
            )
            w_h, w_l = self._plot_weather(
                ax2,
                df_w,
                weather_cols or [],
                kind=weather_kind,
                plot_kwargs=weather_plot_kwargs or {},
            )

            self._create_legends(ax1, ax2, ag_h, ag_l, w_h, w_l)
            self._finalize(
                fig, ax1, title or f"{self.name} Crop/Weather Overlay", save_path
            )
            return fig, ax1, ax2

    def _plot_agri(
        self,
        ax: plt.Axes,
        stats: pd.DataFrame,
        cols: List[str],
        *,
        kind: str = "errorbar",
        plot_kwargs: Dict[str, Any] | None = None,
    ) -> Tuple[List[Any], List[str]]:
        cmap = plt.cm.get_cmap(self.user_cmap, len(cols))
        cmap_agri = plt.cm.get_cmap("Dark2", len(cols))
        handles, labels = [], []
        plot_kwargs = plot_kwargs or {}

        if kind == "bar":
            xnum = mdates.date2num(stats.index.to_pydatetime())
            width = (np.min(np.diff(xnum)) if len(xnum) > 1 else 1.0) * 0.8

        for i, c in enumerate(cols):
            x = stats.index
            y = stats[f"{c}_mean"]
            yerr = stats[f"{c}_std"].fillna(0)

            kwargs = {**self.line_defaults, **plot_kwargs}
            kwargs.setdefault("color", cmap_agri(i))
            kwargs.setdefault("label", c)

            if kind == "errorbar":
                h = ax.errorbar(x, y, yerr=yerr, fmt="o-", capsize=4, **kwargs)
            elif kind == "bar":
                h = ax.bar(x, y, width=width, yerr=yerr, capsize=4, **kwargs)
            elif kind == "line":
                (h,) = ax.plot(x, y, **kwargs)
            elif kind == "scatter":
                h = ax.scatter(x, y, **kwargs)
            else:
                raise ValueError(f"Unknown agri kind: {kind}")

            handles.append(h)
            labels.append(c)

        if kind == "bar":
            ax.xaxis_date()
            ax.figure.autofmt_xdate()

        ax.set_ylabel("Measurements (cm, mm)")
        return handles, labels

    def _plot_weather(
        self,
        ax: plt.Axes,
        df_w: pd.DataFrame,
        cols: List[str],
        *,
        kind: str = "line",
        plot_kwargs: Dict[str, Any] | None = None,
    ) -> Tuple[List[Any], List[str]]:
        cmap = plt.cm.get_cmap(self.user_cmap, len(cols))
        cmap_weather = plt.cm.get_cmap("tab20", len(cols))
        handles, labels = [], []
        plot_kwargs = plot_kwargs or {}

        if kind == "bar":
            xnum = mdates.date2num(df_w.index.to_pydatetime())
            width = (np.min(np.diff(xnum)) if len(xnum) > 1 else 1.0) * 0.8

        for i, code in enumerate(cols):
            x = df_w.index
            y = df_w[code]

            kwargs = {**self.line_defaults, **plot_kwargs}
            kwargs.setdefault("color", cmap_weather(i))
            kwargs.setdefault("label", self.PARAMS.get(code, code))
            kwargs.setdefault("alpha", 0.7)

            if kind == "line":
                (h,) = ax.plot(x, y, **kwargs)
            elif kind == "scatter":
                h = ax.scatter(x, y, s=20, **kwargs)
            elif kind == "bar":
                h = ax.bar(x, y, width=width, **kwargs)
            else:
                raise ValueError(f"Unknown weather kind: {kind}")

            handles.append(h)
            labels.append(self.PARAMS.get(code, code))

        if kind == "bar":
            ax.xaxis_date()
            ax.figure.autofmt_xdate()

        ax.set_ylabel(" / ".join(self.PARAMS.get(c, c) for c in cols))
        return handles, labels

    def _prepare_agri_stats(self, df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        grp = df.groupby("date")[cols].agg(["mean", "std"])
        grp.columns = [f"{col}_{stat}" for col, stat in grp.columns]
        return grp

    def _prepare_weather(self, weather_cols: Optional[List[str]]) -> pd.DataFrame:
        df_w = self.client.load_df().copy()
        df_w["date"] = pd.to_datetime(df_w["date"])
        df_w.set_index("date", inplace=True)
        if weather_cols is None:
            weather_cols = list(df_w.columns)
        missing = set(weather_cols) - set(df_w.columns)
        if missing:
            raise KeyError(f"Weather codes not found: {missing}")
        return df_w[weather_cols]

    def _create_legends(
        self,
        ax1: plt.Axes,
        ax2: plt.Axes,
        ag_handles: List[Any],
        ag_labels: List[str],
        w_handles: List[Any],
        w_labels: List[str],
    ) -> None:
        leg1 = ax1.legend(ag_handles, ag_labels, title="Agri Data", loc="upper left")
        ax1.add_artist(leg1)
        ax2.legend(w_handles, w_labels, title="Weather Data", loc="upper right")

    def _finalize(
        self,
        fig: plt.Figure,
        ax1: plt.Axes,
        title: Optional[str],
        save_path: Optional[str],
        grid: bool = False,
    ) -> None:
        ax1.set_xlabel("Date")
        ax1.set_title(title or f"{self.name} Crop/Weather Overlay")
        ax1.grid(grid)
        fig.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, dpi=300)
        plt.show()
