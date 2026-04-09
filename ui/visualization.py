"""
visualization.py  — Unified single-window AI Search Visualizer
Left panel : all controls (mode, algo, heuristic, nodes, grid settings, speed)
Right panel: matplotlib canvas (graph + search tree + metrics)
User can change any setting and click Run without restarting the program.

Warm palette inspired by: #C9CBA3 · #FFE1A8 · #E26D5C · #723046 · #472D30
"""

import tkinter as tk
from tkinter import ttk
import threading
import time

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import networkx as nx

# ── Warm palette ──────────────────────────────────────────────────────────────
WM_BG      = "#f7f3ee"          # warm off-white background
WM_PANEL   = "#ede8e0"          # slightly darker panel
WM_CARD    = "#ffffff"          # card white
WM_BORDER  = "#d6cfc4"          # warm border
WM_SAGE    = "#C9CBA3"          # sage green  (accent light)
WM_CREAM   = "#FFE1A8"          # cream yellow
WM_TERRA   = "#E26D5C"          # terracotta  (primary action)
WM_PLUM    = "#723046"          # deep plum   (selected)
WM_DARK    = "#472D30"          # darkest     (text)
WM_TEXT    = "#3a2a2d"          # body text
WM_MUTED   = "#9e8e85"          # muted text
WM_HOVER   = "#d45a49"          # terra hover

# ── Search cell colors (warm theme) ──────────────────────────────────────────
COLORS = {
    "empty"    : "#faf8f5",
    "obstacle" : "#472D30",
    "visited"  : "#b8c9a3",   # muted sage green
    "frontier" : "#f5d08a",   # warm amber
    "current"  : "#E26D5C",   # terracotta
    "start"    : "#723046",   # deep plum
    "goal"     : "#c0392b",   # strong red
    "path"     : "#c07c2a",   # golden orange — clearly distinct from frontier
    "grid_line": "#e0d9d0",
}

CITIES = sorted([
    "Abilene","Andover","Anthony","Argonia","Attica","Augusta",
    "Bluff_City","Caldwell","Cheney","Clearwater","Coldwater","Derby",
    "El_Dorado","Emporia","Florence","Greensburg","Harper","Haven",
    "Hillsboro","Hutchinson","Junction_City","Kingman","Kiowa","Leon",
    "Lyons","Manhattan","Marion","Mayfield","McPherson","Medicine_Lodge",
    "Mulvane","Newton","Oxford","Pratt","Rago","Salina","Sawyer",
    "South_Haven","Topeka","Towanda","Viola","Wellington","Wichita",
    "Winfield","Zenda",
])

GRID_HEURISTICS = ["manhattan", "euclidean", "chebyshev"]
CITY_HEURISTICS = ["euclidean_coords", "haversine", "euclidean"]

ALGOS = [
    ("A*",     "astar"),
    ("Greedy", "greedy"),
    ("BFS",    "bfs"),
    ("DFS",    "dfs"),
    ("IDDFS",  "iddfs"),
]


# ═════════════════════════════════════════════════════════════════════════════
#  Main App Window
# ═════════════════════════════════════════════════════════════════════════════

class Visualizer:
    """
    Single-window visualizer.
    Left: control panel (Tkinter widgets)
    Right: matplotlib canvas (graph + tree + metrics)
    """

    def __init__(self, graph_loader_fn, grid_factory_fn):
        """
        graph_loader_fn : callable() → (graph_obj, coords)   for city mode
        grid_factory_fn : callable(rows, cols, obs, conn) → GridEnvironment
        """
        self._load_city   = graph_loader_fn
        self._make_grid   = grid_factory_fn

        self._search_thread = None
        self._running       = False
        self._paused        = False
        self._step_event    = threading.Event()
        self._env           = None
        self._city_pos      = None
        self._graph_obj     = None
        self._bench_needs_restore = False
        self._final_state   = None
        self._final_path    = []

        # ── Root window ───────────────────────────────────────────────────
        self.root = tk.Tk()
        self.root.title("AI Search Visualizer")
        self.root.configure(bg=WM_BG)
        self.root.resizable(True, True)

        W, H = 1280, 780
        sw, sh = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        self.root.geometry(f"{W}x{H}+{(sw-W)//2}+{(sh-H)//2}")

        # ── Layout: left panel + right canvas ─────────────────────────────
        self._build_left_panel()
        self._build_right_canvas()

        # Initial blank draw
        self._blank_canvas()
        self.root.mainloop()

    # ══════════════════════════════════════════════════════════════════════
    #  LEFT PANEL
    # ══════════════════════════════════════════════════════════════════════

    def _build_left_panel(self):
        lp = tk.Frame(self.root, bg=WM_PANEL, width=260)
        lp.pack(side=tk.LEFT, fill=tk.Y)
        lp.pack_propagate(False)
        self._lp = lp

        # Scrollable interior
        canvas = tk.Canvas(lp, bg=WM_PANEL, highlightthickness=0, width=258)
        sb     = tk.Scrollbar(lp, orient=tk.VERTICAL, command=canvas.yview)
        canvas.configure(yscrollcommand=sb.set)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        interior = tk.Frame(canvas, bg=WM_PANEL)
        win_id   = canvas.create_window((0, 0), window=interior, anchor="nw")

        def _on_resize(e):
            canvas.itemconfig(win_id, width=e.width)
        canvas.bind("<Configure>", _on_resize)

        interior.bind("<Configure>",
                      lambda e: canvas.configure(
                          scrollregion=canvas.bbox("all")))

        def _on_mousewheel(event):
            if event.num == 4:
                canvas.yview_scroll(-1, "units")
            elif event.num == 5:
                canvas.yview_scroll(1, "units")
            else:
                canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        def _bind_children(widget):
            widget.bind("<MouseWheel>", _on_mousewheel)
            widget.bind("<Button-4>",   _on_mousewheel)
            widget.bind("<Button-5>",   _on_mousewheel)
            for child in widget.winfo_children():
                _bind_children(child)

        canvas.bind("<MouseWheel>",   _on_mousewheel)
        canvas.bind("<Button-4>",     _on_mousewheel)
        canvas.bind("<Button-5>",     _on_mousewheel)
        interior.bind("<MouseWheel>", _on_mousewheel)
        interior.bind("<Button-4>",   _on_mousewheel)
        interior.bind("<Button-5>",   _on_mousewheel)
        interior.bind("<Map>", lambda e: _bind_children(interior))

        self._build_controls(interior)

    def _build_controls(self, p):
        PAD = dict(padx=16)

        # ── App title ─────────────────────────────────────────────────────
        tk.Label(p, text="AI Search",
                 font=("Georgia", 17, "bold"),
                 bg=WM_PANEL, fg=WM_DARK).pack(anchor="w", padx=16, pady=(20, 0))
        tk.Label(p, text="Visualizer  ·  Part 2",
                 font=("Georgia", 10, "italic"),
                 bg=WM_PANEL, fg=WM_MUTED).pack(anchor="w", padx=16, pady=(0, 12))
        tk.Frame(p, bg=WM_BORDER, height=1).pack(fill=tk.X, padx=16, pady=(0, 14))

        # ── Graph mode ────────────────────────────────────────────────────
        self._section(p, "Graph Mode")
        self.mode_var = tk.StringVar(value="grid")
        mf = tk.Frame(p, bg=WM_PANEL)
        mf.pack(fill=tk.X, **PAD, pady=(4, 10))
        self._mode_btns = {}
        for val, label in [("grid", "⊞  Grid World"), ("city", "◉  City Graph")]:
            b = tk.Button(mf, text=label, bg=WM_CARD, fg=WM_TEXT,
                          font=("DejaVu Sans", 9), relief=tk.FLAT,
                          bd=0, padx=10, pady=6, cursor="hand2",
                          highlightthickness=1,
                          highlightbackground=WM_BORDER,
                          command=lambda v=val: self._set_mode(v))
            b.pack(side=tk.LEFT, padx=(0, 6))
            self._mode_btns[val] = b
        self._set_mode("grid", init=True)

        # ── Algorithm ─────────────────────────────────────────────────────
        self._section(p, "Algorithm")
        self.algo_var = tk.StringVar(value="astar")
        self._algo_btns = {}
        af = tk.Frame(p, bg=WM_PANEL)
        af.pack(fill=tk.X, **PAD, pady=(4, 10))
        for i, (label, val) in enumerate(ALGOS):
            b = tk.Button(af, text=label, bg=WM_CARD, fg=WM_TEXT,
                          font=("DejaVu Sans", 9), relief=tk.FLAT,
                          bd=0, padx=8, pady=6, cursor="hand2",
                          highlightthickness=1,
                          highlightbackground=WM_BORDER,
                          command=lambda v=val: self._set_algo(v))
            b.grid(row=i//3, column=i%3, padx=3, pady=3, sticky="ew")
            af.columnconfigure(i%3, weight=1)
            self._algo_btns[val] = b
        self._set_algo("astar", init=True)

        # ── Heuristic ─────────────────────────────────────────────────────
        self._section(p, "Heuristic")
        self.heuristic_var = tk.StringVar(value="manhattan")
        self.h_combo = ttk.Combobox(p, textvariable=self.heuristic_var,
                                    values=GRID_HEURISTICS, state="readonly",
                                    width=22, font=("DejaVu Sans", 9))
        self._style_combo(self.h_combo)
        self.h_combo.pack(fill=tk.X, **PAD, pady=(4, 10))

        # ── Placeholder slot — city/grid sections pack into here, not into p ──
        # This guarantees they appear between heuristic and speed, always.
        self._mode_slot = tk.Frame(p, bg=WM_PANEL)
        self._mode_slot.pack(fill=tk.X)

        # ── City nodes ────────────────────────────────────────────────────
        self.city_section = tk.Frame(self._mode_slot, bg=WM_PANEL)
        self._section(self.city_section, "Graph Layout")
        self.layout_var = tk.StringVar(value="spring")
        layout_frame = tk.Frame(self.city_section, bg=WM_PANEL)
        layout_frame.pack(fill=tk.X, padx=16, pady=(4, 10))
        self._layout_btns = {}
        for val, txt in [("geographic", "Geographic"), ("spring", "Spring")]:
            b = tk.Button(layout_frame, text=txt, bg=WM_CARD, fg=WM_TEXT,
                          font=("DejaVu Sans", 8), relief=tk.FLAT,
                          bd=0, padx=6, pady=5, cursor="hand2",
                          highlightthickness=1, highlightbackground=WM_BORDER,
                          command=lambda v=val: self._set_layout(v))
            b.pack(side=tk.LEFT, padx=(0, 4))
            self._layout_btns[val] = b
        self._set_layout("spring", init=True)

        self._section(self.city_section, "Start City")
        self.start_var = tk.StringVar(value="Wichita")
        sc = ttk.Combobox(self.city_section, textvariable=self.start_var,
                          values=CITIES, state="readonly",
                          width=22, font=("DejaVu Sans", 9))
        self._style_combo(sc)
        sc.pack(fill=tk.X, padx=16, pady=(4, 8))

        self._section(self.city_section, "Goal City")
        self.goal_var = tk.StringVar(value="Topeka")
        gc = ttk.Combobox(self.city_section, textvariable=self.goal_var,
                          values=CITIES, state="readonly",
                          width=22, font=("DejaVu Sans", 9))
        self._style_combo(gc)
        gc.pack(fill=tk.X, padx=16, pady=(4, 10))

        # ── Grid settings ─────────────────────────────────────────────────
        self.grid_section = tk.Frame(self._mode_slot, bg=WM_PANEL)
        self._section(self.grid_section, "Grid Size")
        gsz = tk.Frame(self.grid_section, bg=WM_PANEL)
        gsz.pack(fill=tk.X, padx=16, pady=(4, 6))

        self.rows_var = tk.IntVar(value=10)
        self.cols_var = tk.IntVar(value=10)
        for lbl, var in [("Rows", self.rows_var), ("Cols", self.cols_var)]:
            f = tk.Frame(gsz, bg=WM_PANEL)
            f.pack(side=tk.LEFT, padx=(0, 12))
            tk.Label(f, text=lbl, font=("DejaVu Sans", 8),
                     bg=WM_PANEL, fg=WM_MUTED).pack(anchor="w")
            tk.Spinbox(f, from_=5, to=30, textvariable=var, width=5,
                       font=("DejaVu Sans", 10, "bold"),
                       bg=WM_CARD, fg=WM_DARK,
                       relief=tk.FLAT, bd=2,
                       buttonbackground=WM_BORDER).pack()

        self._section(self.grid_section, "Obstacle Density")
        self.obs_var = tk.DoubleVar(value=0.25)
        obs_f = tk.Frame(self.grid_section, bg=WM_PANEL)
        obs_f.pack(fill=tk.X, padx=16, pady=(4, 6))
        self.obs_label = tk.Label(obs_f, text="25%",
                                  font=("DejaVu Sans", 9, "bold"),
                                  bg=WM_PANEL, fg=WM_TERRA, width=4)
        self.obs_label.pack(side=tk.RIGHT)
        obs_sl = tk.Scale(obs_f, variable=self.obs_var,
                          from_=0.20, to=0.30, resolution=0.01,
                          orient=tk.HORIZONTAL, showvalue=False,
                          bg=WM_PANEL, fg=WM_DARK,
                          troughcolor=WM_CREAM, activebackground=WM_TERRA,
                          highlightthickness=0, sliderrelief=tk.FLAT,
                          command=lambda v: self.obs_label.configure(
                              text=f"{float(v)*100:.0f}%"))
        obs_sl.pack(fill=tk.X)

        self._section(self.grid_section, "Connectivity")
        self.conn_var = tk.IntVar(value=4)
        cf = tk.Frame(self.grid_section, bg=WM_PANEL)
        cf.pack(fill=tk.X, padx=16, pady=(4, 10))
        self._conn_btns = {}
        for val, txt in [(4, "4-way"), (8, "8-way")]:
            b = tk.Button(cf, text=txt, bg=WM_CARD, fg=WM_TEXT,
                          font=("DejaVu Sans", 9), relief=tk.FLAT,
                          bd=0, padx=10, pady=5, cursor="hand2",
                          highlightthickness=1,
                          highlightbackground=WM_BORDER,
                          command=lambda v=val: self._set_conn(v))
            b.pack(side=tk.LEFT, padx=(0, 6))
            self._conn_btns[val] = b
        self._set_conn(4, init=True)

        # Quick presets — inside grid_section so they appear after heuristic
        self._section(self.grid_section, "Quick Preset")
        self._quick_preset_frame = tk.Frame(self.grid_section, bg=WM_PANEL)
        self._quick_preset_frame.pack(fill=tk.X, padx=16, pady=(4, 10))
        for label, rows, cols, obs in [
            ("Easy",   8,  8,  0.20),
            ("Medium", 15, 15, 0.25),
            ("Hard",   25, 25, 0.30),
        ]:
            b = tk.Button(self._quick_preset_frame, text=label, bg=WM_CARD, fg=WM_TEXT,
                          font=("DejaVu Sans", 8), relief=tk.FLAT,
                          bd=0, padx=6, pady=4, cursor="hand2",
                          highlightthickness=1, highlightbackground=WM_BORDER,
                          command=lambda r=rows, c=cols, o=obs: self._apply_quick_preset(r, c, o))
            b.pack(side=tk.LEFT, padx=(0, 4))

        # ── Mode-specific section (city or grid) goes here, in correct order ──
        # Both sections are children of p but only one is packed at a time.
        # _refresh_mode_sections() controls which is visible.
        self._refresh_mode_sections()

        # ── Speed ─────────────────────────────────────────────────────────
        tk.Frame(p, bg=WM_BORDER, height=1).pack(fill=tk.X, padx=16, pady=(6, 10))
        self._section(p, "Animation Speed")
        spd_f = tk.Frame(p, bg=WM_PANEL)
        spd_f.pack(fill=tk.X, padx=16, pady=(4, 10))
        tk.Label(spd_f, text="Fast", font=("DejaVu Sans", 8),
                 bg=WM_PANEL, fg=WM_MUTED).pack(side=tk.LEFT)
        self.speed_var = tk.DoubleVar(value=0.15)
        tk.Scale(spd_f, variable=self.speed_var,
                 from_=0.02, to=1.0, resolution=0.01,
                 orient=tk.HORIZONTAL, showvalue=False,
                 bg=WM_PANEL, troughcolor=WM_CREAM,
                 activebackground=WM_TERRA,
                 highlightthickness=0, sliderrelief=tk.FLAT).pack(
                     side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        tk.Label(spd_f, text="Slow", font=("DejaVu Sans", 8),
                 bg=WM_PANEL, fg=WM_MUTED).pack(side=tk.LEFT)

        # ── Control buttons ───────────────────────────────────────────────
        tk.Frame(p, bg=WM_BORDER, height=1).pack(fill=tk.X, padx=16, pady=(2, 12))

        # ── View toggle (Search / Benchmark) ──────────────────────────────
        self._view_var = tk.StringVar(value="search")
        vf = tk.Frame(p, bg=WM_PANEL)
        vf.pack(fill=tk.X, padx=16, pady=(0, 10))
        self._view_btns = {}
        for val, label in [("search", "⬡  Search"), ("benchmark", "⧖  Benchmark")]:
            b = tk.Button(vf, text=label, bg=WM_CARD, fg=WM_TEXT,
                          font=("DejaVu Sans", 8), relief=tk.FLAT,
                          bd=0, padx=8, pady=5, cursor="hand2",
                          highlightthickness=1, highlightbackground=WM_BORDER,
                          command=lambda v=val: self._set_view(v))
            b.pack(side=tk.LEFT, padx=(0, 4))
            self._view_btns[val] = b

        # ── Search controls ───────────────────────────────────────────────
        self._search_controls = tk.Frame(p, bg=WM_PANEL)
        self._search_controls.pack(fill=tk.X)

        self.run_btn = self._action_btn(self._search_controls, "▶  Run", self._on_run,
                                        WM_TERRA, "white")
        self.pause_btn = self._action_btn(self._search_controls, "⏸  Pause", self._on_pause,
                                          WM_PLUM, "white")
        self._action_btn(self._search_controls, "↺  Reset", self._on_reset, WM_SAGE, WM_DARK)

        # ── Benchmark controls ────────────────────────────────────────────
        self._bench_controls = tk.Frame(p, bg=WM_PANEL)

        # Grid benchmark settings
        self._bench_grid_section = tk.Frame(self._bench_controls, bg=WM_PANEL)
        self._section(self._bench_grid_section, "Complexity Presets")
        self._bench_preset_var = tk.StringVar(value="all")
        preset_f = tk.Frame(self._bench_grid_section, bg=WM_PANEL)
        preset_f.pack(fill=tk.X, padx=16, pady=(4, 6))
        self._preset_btns = {}
        for val, txt in [("all", "Easy+Med+Hard"), ("custom", "Custom")]:
            b = tk.Button(preset_f, text=txt, bg=WM_CARD, fg=WM_TEXT,
                          font=("DejaVu Sans", 8), relief=tk.FLAT,
                          bd=0, padx=6, pady=4, cursor="hand2",
                          highlightthickness=1, highlightbackground=WM_BORDER,
                          command=lambda v=val: self._set_bench_preset(v))
            b.pack(side=tk.LEFT, padx=(0, 4))
            self._preset_btns[val] = b

        # Custom grid settings (shown when "Custom" selected)
        self._bench_custom = tk.Frame(self._bench_grid_section, bg=WM_PANEL)
        self._section(self._bench_custom, "Custom Grid")
        csz = tk.Frame(self._bench_custom, bg=WM_PANEL)
        csz.pack(fill=tk.X, padx=16, pady=(4, 4))
        self._bench_rows = tk.IntVar(value=10)
        self._bench_cols = tk.IntVar(value=10)
        self._bench_obs  = tk.DoubleVar(value=0.25)
        self._bench_runs = tk.IntVar(value=5)
        for lbl, var, lo, hi in [
            ("Rows", self._bench_rows, 5, 30),
            ("Cols", self._bench_cols, 5, 30),
        ]:
            f = tk.Frame(csz, bg=WM_PANEL)
            f.pack(side=tk.LEFT, padx=(0, 10))
            tk.Label(f, text=lbl, font=("DejaVu Sans", 7),
                     bg=WM_PANEL, fg=WM_MUTED).pack(anchor="w")
            tk.Spinbox(f, from_=lo, to=hi, textvariable=var, width=4,
                       font=("DejaVu Sans", 9, "bold"),
                       bg=WM_CARD, fg=WM_DARK, relief=tk.FLAT, bd=2,
                       buttonbackground=WM_BORDER).pack()

        obs_f = tk.Frame(self._bench_custom, bg=WM_PANEL)
        obs_f.pack(fill=tk.X, padx=16, pady=(2, 4))
        self._bench_obs_label = tk.Label(obs_f, text="25%",
                                          font=("DejaVu Sans", 8, "bold"),
                                          bg=WM_PANEL, fg=WM_TERRA, width=4)
        self._bench_obs_label.pack(side=tk.RIGHT)
        tk.Label(obs_f, text="Obstacle %", font=("DejaVu Sans", 7),
                 bg=WM_PANEL, fg=WM_MUTED).pack(anchor="w")
        tk.Scale(obs_f, variable=self._bench_obs,
                 from_=0.20, to=0.30, resolution=0.01,
                 orient=tk.HORIZONTAL, showvalue=False,
                 bg=WM_PANEL, troughcolor=WM_CREAM, activebackground=WM_TERRA,
                 highlightthickness=0, sliderrelief=tk.FLAT,
                 command=lambda v: self._bench_obs_label.configure(
                     text=f"{float(v)*100:.0f}%")).pack(fill=tk.X)

        runs_f = tk.Frame(self._bench_custom, bg=WM_PANEL)
        runs_f.pack(fill=tk.X, padx=16, pady=(2, 6))
        tk.Label(runs_f, text="Runs", font=("DejaVu Sans", 7),
                 bg=WM_PANEL, fg=WM_MUTED).pack(side=tk.LEFT)
        tk.Spinbox(runs_f, from_=3, to=20, textvariable=self._bench_runs, width=4,
                   font=("DejaVu Sans", 9, "bold"),
                   bg=WM_CARD, fg=WM_DARK, relief=tk.FLAT, bd=2,
                   buttonbackground=WM_BORDER).pack(side=tk.LEFT, padx=(6, 0))

        self._bench_grid_section.pack(fill=tk.X)
        self._set_view("search", init=True)
        self._bench_controls.pack_forget()
        self._set_bench_preset("all", init=True)

        # City benchmark — no extra dropdowns, reuses start_var / goal_var from search
        self._bench_city_section = tk.Frame(self._bench_controls, bg=WM_PANEL)
        tk.Label(self._bench_city_section,
                 text="Uses Start / Goal cities selected above.",
                 font=("DejaVu Sans", 8, "italic"),
                 bg=WM_PANEL, fg=WM_MUTED, wraplength=210).pack(
                     padx=16, pady=(8, 4), anchor="w")

        self._action_btn(self._bench_controls, "▶  Run Benchmark",
                         self._on_benchmark, WM_TERRA, "white")

        self._bench_status_var = tk.StringVar(value="")
        tk.Label(self._bench_controls, textvariable=self._bench_status_var,
                 font=("DejaVu Sans", 7, "italic"),
                 bg=WM_PANEL, fg=WM_MUTED, wraplength=220).pack(padx=16, pady=(4, 4))

        # ── Close pinned to very bottom ────────────────────────────────────
        bottom_frame = tk.Frame(p, bg=WM_PANEL)
        bottom_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=(0, 16))

        self.status_var = tk.StringVar(value="Ready")
        tk.Label(bottom_frame, textvariable=self.status_var,
                 font=("DejaVu Sans", 8, "italic"),
                 bg=WM_PANEL, fg=WM_MUTED,
                 wraplength=220).pack(padx=16, pady=(4, 4))

        close_b = tk.Button(bottom_frame, text="✕  Close",
                            command=self._on_close,
                            bg="#6b2d2d", fg="white",
                            activebackground="#6b2d2d", activeforeground="white",
                            font=("DejaVu Sans", 10, "bold"),
                            relief=tk.FLAT, bd=0, padx=0, pady=9, cursor="hand2")
        close_b.pack(fill=tk.X, padx=16, pady=(0, 16))
        close_b.bind("<Enter>", lambda e: close_b.configure(bg=_darken("#6b2d2d")))
        close_b.bind("<Leave>", lambda e: close_b.configure(bg="#6b2d2d"))

    def _section(self, parent, text):
        tk.Label(parent, text=text.upper(),
                 font=("DejaVu Sans", 7, "bold"),
                 bg=parent.cget("bg"), fg=WM_MUTED).pack(
                     anchor="w", padx=16, pady=(8, 0))

    def _action_btn(self, parent, text, cmd, bg, fg):
        b = tk.Button(parent, text=text, command=cmd,
                      bg=bg, fg=fg, activebackground=bg,
                      activeforeground=fg,
                      font=("DejaVu Sans", 10, "bold"),
                      relief=tk.FLAT, bd=0,
                      padx=0, pady=9, cursor="hand2")
        b.pack(fill=tk.X, padx=16, pady=3)
        b.bind("<Enter>", lambda e, b=b, c=bg: b.configure(bg=_darken(c)))
        b.bind("<Leave>", lambda e, b=b, c=bg: b.configure(bg=c))
        return b

    def _style_combo(self, cb):
        s = ttk.Style()
        s.theme_use("clam")
        s.configure("Warm.TCombobox",
                    fieldbackground=WM_CARD, background=WM_BORDER,
                    foreground=WM_TEXT, selectbackground=WM_TERRA,
                    selectforeground="white", bordercolor=WM_BORDER,
                    lightcolor=WM_BORDER, darkcolor=WM_BORDER,
                    arrowcolor=WM_TERRA)
        s.map("Warm.TCombobox",
              fieldbackground=[("readonly", WM_CARD)],
              foreground=[("readonly", WM_TEXT)])
        cb.configure(style="Warm.TCombobox")

    # ── Mode / algo / conn toggle helpers ────────────────────────────────

    def _set_mode(self, val, init=False):
        self.mode_var.set(val)
        for v, b in self._mode_btns.items():
            b.configure(bg=WM_TERRA if v == val else WM_CARD,
                        fg="white" if v == val else WM_TEXT,
                        highlightbackground=WM_TERRA if v == val else WM_BORDER)
        if not init:
            self._refresh_mode_sections()
            if self._view_var.get() == "benchmark":
                self._refresh_bench_sections()

    def _refresh_mode_sections(self):
        mode = self.mode_var.get()
        if mode == "grid":
            self.city_section.pack_forget()
            self.grid_section.pack(fill=tk.X, in_=self._mode_slot)
            self.h_combo.configure(values=GRID_HEURISTICS)
            if self.heuristic_var.get() not in GRID_HEURISTICS:
                self.heuristic_var.set("manhattan")
        else:
            self.grid_section.pack_forget()
            self.city_section.pack(fill=tk.X, in_=self._mode_slot)
            self.h_combo.configure(values=CITY_HEURISTICS)
            if self.heuristic_var.get() not in CITY_HEURISTICS:
                self.heuristic_var.set("euclidean_coords")

    def _apply_quick_preset(self, rows, cols, obs):
        """Set grid size and obstacle density from a quick preset and run."""
        self.rows_var.set(rows)
        self.cols_var.set(cols)
        self.obs_var.set(obs)
        self.obs_label.configure(text=f"{int(obs*100)}%")
        self._on_run()

    def _set_algo(self, val, init=False):
        self.algo_var.set(val)
        for v, b in self._algo_btns.items():
            b.configure(bg=WM_PLUM if v == val else WM_CARD,
                        fg="white" if v == val else WM_TEXT,
                        highlightbackground=WM_PLUM if v == val else WM_BORDER)

    def _set_layout(self, val, init=False):
        self.layout_var.set(val)
        for v, b in self._layout_btns.items():
            b.configure(bg=WM_TERRA if v == val else WM_CARD,
                        fg="white" if v == val else WM_TEXT)
        if not init:
            self._graph_obj = None  # force recompute layout on next run

    def _set_conn(self, val, init=False):
        self.conn_var.set(val)
        for v, b in self._conn_btns.items():
            b.configure(bg=WM_TERRA if v == val else WM_CARD,
                        fg="white" if v == val else WM_TEXT)

    def _set_view(self, val, init=False):
        self._view_var.set(val)
        for v, b in self._view_btns.items():
            b.configure(bg=WM_PLUM if v == val else WM_CARD,
                        fg="white" if v == val else WM_TEXT,
                        highlightbackground=WM_PLUM if v == val else WM_BORDER)
        if not init:
            if val == "search":
                self._bench_controls.pack_forget()
                self._search_controls.pack(fill=tk.X)
                # Restore original 2x2 GridSpec layout
                if getattr(self, "_bench_needs_restore", False):
                    self.fig.clear()
                    gs = gridspec.GridSpec(2, 2, figure=self.fig,
                                          height_ratios=[5, 1.4],
                                          width_ratios=[7, 3],
                                          hspace=0.35, wspace=0.3)
                    self.ax_main    = self.fig.add_subplot(gs[0, 0])
                    self.ax_tree    = self.fig.add_subplot(gs[0, 1])
                    self.ax_metrics = self.fig.add_subplot(gs[1, :])
                    self._bench_needs_restore = False
                self._blank_canvas()
            else:
                self._running = False
                self._search_controls.pack_forget()
                self._bench_controls.pack(fill=tk.X)
                self._refresh_bench_sections()
                self._blank_bench_canvas()

    def _refresh_bench_sections(self):
        mode = self.mode_var.get()
        if mode == "grid":
            self._bench_city_section.pack_forget()
            self._bench_grid_section.pack(fill=tk.X)
        else:
            self._bench_grid_section.pack_forget()
            self._bench_city_section.pack(fill=tk.X)

    def _set_bench_preset(self, val, init=False):
        self._bench_preset_var.set(val)
        for v, b in self._preset_btns.items():
            b.configure(bg=WM_TERRA if v == val else WM_CARD,
                        fg="white" if v == val else WM_TEXT)
        if not init:
            if val == "custom":
                self._bench_custom.pack(fill=tk.X)
            else:
                self._bench_custom.pack_forget()

    def _blank_bench_canvas(self):
        for ax in [self.ax_main, self.ax_tree, self.ax_metrics]:
            ax.cla()
            ax.set_facecolor(WM_BG)
            ax.axis("off")
        self.ax_main.text(0.5, 0.5,
                          "Configure benchmark settings\nand click  Run Benchmark",
                          ha="center", va="center",
                          fontsize=13, color=WM_MUTED,
                          fontfamily="DejaVu Serif",
                          transform=self.ax_main.transAxes)
        self.ax_main.set_title("Benchmark Results", color=WM_DARK, fontsize=11, pad=8)
        self.canvas.draw()

    def _on_benchmark(self):
        if self._running:
            return
        mode = self.mode_var.get()
        self._bench_status_var.set("Running benchmark…")
        self.root.update_idletasks()

        thread = threading.Thread(
            target=self._run_benchmark, args=(mode,), daemon=True)
        thread.start()

    def _run_benchmark(self, mode):
        from benchmark.benchmark import batch_compare, plot_complexity_chart
        from core.heuristics import get_heuristic
        import numpy as np

        try:
            if mode == "grid":
                from core.grid import GridEnvironment
                preset = self._bench_preset_var.get()

                if preset == "all":
                    settings = [
                        {"label": "easy",   "rows": 8,  "cols": 8,  "obs": 0.20, "conn": 4},
                        {"label": "medium", "rows": 15, "cols": 15, "obs": 0.25, "conn": 4},
                        {"label": "hard",   "rows": 25, "cols": 25, "obs": 0.30, "conn": 4},
                    ]
                else:
                    settings = [{"label": "custom",
                                 "rows": self._bench_rows.get(),
                                 "cols": self._bench_cols.get(),
                                 "obs":  self._bench_obs.get(),
                                 "conn": self.conn_var.get()}]

                all_results = {}
                for s in settings:
                    def env_factory(seed, s=s):
                        return GridEnvironment(
                            rows=s["rows"], cols=s["cols"],
                            obstacle_pct=s["obs"], connectivity=s["conn"],
                            seed=seed)
                    n = self._bench_runs.get() if preset == "custom" else 5
                    results = batch_compare(env_factory, n_runs=n, label=s["label"])
                    all_results[s["label"]] = results

                self.root.after(0, self._draw_bench_chart, all_results)

            else:
                # City mode — run all algos on selected start/goal
                if self._graph_obj is None:
                    self._graph_obj = self._load_city()
                graph_obj = self._graph_obj
                start = self.start_var.get()
                goal  = self.goal_var.get()

                if start == goal:
                    self.root.after(0, self._bench_status_var.set,
                                    "Start and goal must differ.")
                    return

                from core.search import bfs, dfs, iddfs, greedy_best_first, astar, reconstruct_path
                import time, tracemalloc

                algos = [
                    ("bfs",    lambda g, h: bfs(start, goal, g)),
                    ("dfs",    lambda g, h: dfs(start, goal, g)),
                    ("iddfs",  lambda g, h: iddfs(start, goal, g)),
                    ("greedy", lambda g, h: greedy_best_first(start, goal, g, h)),
                    ("astar",  lambda g, h: astar(start, goal, g, h)),
                ]
                h_fn = get_heuristic(self.heuristic_var.get(),
                                     coords=graph_obj.coords)
                results = []
                for name, fn in algos:
                    tracemalloc.start()
                    t0 = time.perf_counter()
                    final = None
                    expanded = 0
                    for state in fn(graph_obj.adjacency, h_fn):
                        final = state
                        expanded += 1
                        if state.get("found"):
                            break
                    t1 = time.perf_counter()
                    _, peak = tracemalloc.get_traced_memory()
                    tracemalloc.stop()

                    path = []
                    cost = float("inf")
                    if final and final.get("found"):
                        path = reconstruct_path(final["parent"], start, goal)
                        cost = sum(
                            graph_obj.edge_weight(path[i], path[i+1])
                            for i in range(len(path)-1))

                    results.append({
                        "algo":      name,
                        "time_ms":   round((t1-t0)*1000, 3),
                        "mem_kb":    round(peak/1024, 2),
                        "expanded":  expanded,
                        "cost":      round(cost, 3) if cost != float("inf") else None,
                        "found":     bool(final and final.get("found")),
                    })

                self.root.after(0, self._draw_city_bench_chart, results, start, goal)

        except Exception as ex:
            self.root.after(0, self._bench_status_var.set, f"Error: {ex}")

    def _draw_bench_chart(self, all_results):
        """Draw 3-panel bar chart for grid complexity benchmark — equal 1x3 layout."""
        import numpy as np

        from benchmark.benchmark import ALGORITHMS as BENCH_ALGOS
        algo_names = list(BENCH_ALGOS.keys())
        settings   = list(all_results.keys())
        n_algos    = len(algo_names)

        def get_metric(key, fallback=0.0):
            vals = []
            for setting in settings:
                row = []
                for algo in algo_names:
                    match = next((s for s in all_results[setting]
                                  if s.get("algo") == algo), None)
                    row.append(match.get(key, fallback)
                               if match and "note" not in match else fallback)
                vals.append(row)
            return vals

        times    = get_metric("time_mean")
        mems     = get_metric("mem_mean_kb")
        expanded = get_metric("exp_mean")
        tstds    = get_metric("time_std")
        mstds    = get_metric("mem_std_kb")
        estds    = get_metric("exp_std")

        # ── Replace figure with equal 1x3 layout ─────────────────────────
        self.fig.clear()
        axes = self.fig.subplots(1, 3)
        self.fig.subplots_adjust(wspace=0.35, left=0.07, right=0.97,
                                  top=0.88, bottom=0.18)

        x      = np.arange(n_algos)
        bw     = 0.22
        colors = [WM_TERRA, WM_PLUM, WM_SAGE]
        labels = [s.capitalize() for s in settings]

        panels = [
            (axes[0], times,    tstds, "Runtime (ms)",   "Wall-clock Time"),
            (axes[1], mems,     mstds, "Memory (KB)",    "Peak Memory"),
            (axes[2], expanded, estds, "Nodes Expanded", "Search Effort"),
        ]

        for ax, metric_vals, std_vals, ylabel, title in panels:
            ax.set_facecolor(WM_CARD)
            for i, (sv, ss, color, lbl) in enumerate(
                    zip(metric_vals, std_vals, colors, labels)):
                ax.bar(x + (i-1)*bw, sv, bw,
                       label=lbl, color=color, alpha=0.85,
                       yerr=ss, capsize=3,
                       error_kw={"elinewidth": 1, "ecolor": WM_MUTED})
            ax.set_title(title, color=WM_DARK, fontsize=9, fontweight="bold", pad=6)
            ax.set_ylabel(ylabel, fontsize=7, color=WM_MUTED)
            ax.set_xticks(x)
            ax.set_xticklabels([a.upper() for a in algo_names],
                               fontsize=6, rotation=15, color=WM_DARK)
            ax.legend(fontsize=6)
            ax.grid(axis="y", linestyle="--", alpha=0.3)
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(colors=WM_MUTED, length=0)
            ax.set_facecolor(WM_CARD)
            self.fig.patch.set_facecolor(WM_BG)

        self.canvas.draw()
        self.fig.savefig("benchmark_results.png", dpi=100, bbox_inches="tight")
        self._bench_status_var.set("Done! Chart saved → benchmark_results.png")
        self._bench_needs_restore = True

    def _draw_city_bench_chart(self, results, start, goal):
        """Draw bar chart comparing all algos on a single city pair — equal 1x3 layout."""
        import numpy as np

        algos  = [r["algo"] for r in results]
        times  = [r["time_ms"] for r in results]
        mems   = [r["mem_kb"] for r in results]
        exps   = [r["expanded"] for r in results]
        found  = [r["found"] for r in results]
        colors = [WM_TERRA if f else WM_MUTED for f in found]

        # ── Replace figure with equal 1x3 layout ─────────────────────────
        self.fig.clear()
        axes = self.fig.subplots(1, 3)
        self.fig.subplots_adjust(wspace=0.35, left=0.07, right=0.97,
                                  top=0.88, bottom=0.18)
        self.fig.patch.set_facecolor(WM_BG)

        x = np.arange(len(algos))
        panels = [
            (axes[0], times, "Runtime (ms)",   f"{start} → {goal}"),
            (axes[1], mems,  "Memory (KB)",    "Peak Memory"),
            (axes[2], exps,  "Nodes Expanded", "Search Effort"),
        ]

        for ax, vals, ylabel, title in panels:
            ax.set_facecolor(WM_CARD)
            ax.bar(x, vals, color=colors, alpha=0.85, width=0.5)
            ax.set_title(title, color=WM_DARK, fontsize=9,
                         fontweight="bold", pad=6)
            ax.set_ylabel(ylabel, fontsize=7, color=WM_MUTED)
            ax.set_xticks(x)
            ax.set_xticklabels([a.upper() for a in algos],
                               fontsize=7, rotation=15, color=WM_DARK)
            ax.grid(axis="y", linestyle="--", alpha=0.3)
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(colors=WM_MUTED, length=0)

        # Annotate costs on runtime panel
        for i, r in enumerate(results):
            if r["found"] and r["cost"]:
                axes[0].text(i, times[i] + max(times)*0.02,
                             f"cost\n{r['cost']:.1f}",
                             ha="center", va="bottom",
                             fontsize=5.5, color=WM_DARK)

        self.canvas.draw()
        self._bench_status_var.set(
            f"Done!  {start} → {goal}  ·  "
            f"{sum(found)}/{len(found)} algorithms found a path")
        self._bench_needs_restore = True

    # ══════════════════════════════════════════════════════════════════════
    #  RIGHT CANVAS (matplotlib embedded in Tkinter)
    # ══════════════════════════════════════════════════════════════════════

    def _build_right_canvas(self):
        rp = tk.Frame(self.root, bg=WM_BG)
        rp.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.fig = plt.Figure(figsize=(10, 7), facecolor=WM_BG)
        gs = gridspec.GridSpec(2, 2, figure=self.fig,
                               height_ratios=[5, 1.4],
                               width_ratios=[7, 3],
                               hspace=0.35, wspace=0.3)
        self.ax_main    = self.fig.add_subplot(gs[0, 0])
        self.ax_tree    = self.fig.add_subplot(gs[0, 1])
        self.ax_metrics = self.fig.add_subplot(gs[1, :])

        self.canvas = FigureCanvasTkAgg(self.fig, master=rp)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        # ── Tooltip state ─────────────────────────────────────────────────
        self._tooltip_annotation = None
        self._tooltip_pinned     = False   # True after click, stays visible
        self.canvas.mpl_connect("motion_notify_event", self._on_mouse_move)
        self.canvas.mpl_connect("button_press_event",  self._on_mouse_click)

    def _blank_canvas(self):
        for ax in [self.ax_main, self.ax_tree, self.ax_metrics]:
            ax.cla()
            ax.set_facecolor(WM_BG)
            ax.axis("off")
        self.ax_main.text(0.5, 0.5, "Configure settings\nand click  Run",
                          ha="center", va="center",
                          fontsize=13, color=WM_MUTED,
                          fontfamily="DejaVu Serif",
                          transform=self.ax_main.transAxes)
        self.ax_main.set_title("Search View", color=WM_DARK, fontsize=11, pad=8)
        self.ax_tree.set_title("Search Tree", color=WM_DARK, fontsize=11, pad=8)
        self.canvas.draw()

    # ══════════════════════════════════════════════════════════════════════
    #  BUTTON HANDLERS
    # ══════════════════════════════════════════════════════════════════════

    def _on_run(self):
        if self._running:
            self._running = False
            time.sleep(0.1)

        self._paused = False
        self._step_event.set()
        self.pause_btn.configure(text="⏸  Pause")

        # Build environment
        mode = self.mode_var.get()
        try:
            if mode == "grid":
                env = self._make_grid(
                    self.rows_var.get(),
                    self.cols_var.get(),
                    self.obs_var.get(),
                    self.conn_var.get(),
                )
                self._env       = env
                self._city_pos  = None
                graph_type      = "grid"
            else:
                if self._graph_obj is None:
                    self._graph_obj = self._load_city()
                graph_obj  = self._graph_obj
                coords     = graph_obj.coords
                start      = self.start_var.get()
                goal       = self.goal_var.get()
                if start == goal:
                    self._set_status("Start and goal must differ.", error=True)
                    return
                if start not in graph_obj.adjacency or goal not in graph_obj.adjacency:
                    self._set_status("City not found in graph.", error=True)
                    return

                class EnvWrapper:
                    def __init__(self, adj, start, goal, coords):
                        self.adjacency = adj
                        self.start     = start
                        self.goal      = goal
                        self.coords    = coords
                    def edge_weight(self, a, b):
                        for nb, w in self.adjacency.get(a, []):
                            if nb == b: return w
                        return 1.0

                self._env      = EnvWrapper(graph_obj.adjacency, start, goal, coords)
                self._city_pos = self._compute_city_layout(self._env)
                graph_type     = "city"

        except Exception as ex:
            self._set_status(f"Error: {ex}", error=True)
            return

        # Reset search tree
        self._tree = nx.DiGraph()
        self._tree.add_node(self._env.start)
        self._graph_type = graph_type

        # Get search generator
        from core.search import (bfs, dfs, iddfs, greedy_best_first,
                            astar, reconstruct_path)
        from core.heuristics import get_heuristic

        hname   = self.heuristic_var.get()
        coords_ = getattr(self._env, "coords", None)
        h_fn    = get_heuristic(hname, coords_)
        method  = self.algo_var.get()
        env     = self._env

        if method == "bfs":
            gen = bfs(env.start, env.goal, env.adjacency)
        elif method == "dfs":
            gen = dfs(env.start, env.goal, env.adjacency)
        elif method == "iddfs":
            gen = iddfs(env.start, env.goal, env.adjacency)
        elif method == "greedy":
            gen = greedy_best_first(env.start, env.goal, env.adjacency, h_fn)
        elif method == "astar":
            gen = astar(env.start, env.goal, env.adjacency, h_fn)
        else:
            gen = bfs(env.start, env.goal, env.adjacency)

        self._reconstruct = reconstruct_path
        self._running     = True
        self._set_status("Running…")

        self._search_thread = threading.Thread(
            target=self._run_search, args=(gen,), daemon=True)
        self._search_thread.start()

    def _on_pause(self):
        if not self._running:
            return
        self._paused = not self._paused
        if self._paused:
            self.pause_btn.configure(text="▶  Resume")
            self._set_status("Paused")
        else:
            self.pause_btn.configure(text="⏸  Pause")
            self._step_event.set()
            self._set_status("Running…")

    def _on_close(self):
        self._running = False
        self.root.destroy()

    def _on_reset(self):
        self._running     = False
        self._paused      = False
        self._final_state = None
        self._final_path  = []
        self.pause_btn.configure(text="⏸  Pause")
        self._blank_canvas()
        self._set_status("Ready")

    def _set_status(self, msg, error=False):
        self.status_var.set(msg)
        self.status_var._root = self.root
        color = "#c0392b" if error else WM_MUTED
        # find and recolor the status label
        for w in self._lp.winfo_children():
            pass  # status label is updated via StringVar automatically

    # ══════════════════════════════════════════════════════════════════════
    #  SEARCH LOOP  (runs in background thread, draws via root.after)
    # ══════════════════════════════════════════════════════════════════════

    def _run_search(self, gen):
        final_state = None
        for state in gen:
            if not self._running:
                return
            while self._paused:
                time.sleep(0.05)
                if not self._running:
                    return

            final_state = state
            self._update_tree(state)
            self.root.after(0, self._draw, state, [])
            time.sleep(self.speed_var.get())

            if state.get("found"):
                break

        if final_state and final_state.get("found"):
            from core.search import reconstruct_path
            path = reconstruct_path(
                final_state["parent"], self._env.start, self._env.goal)
        else:
            path = []

        self.root.after(0, self._draw, final_state, path)
        self._running     = False
        self._final_state = final_state   # saved for hover/click tooltip
        self._final_path  = path

        if path:
            cost = _path_cost(self._env, path)
            self.root.after(0, self._set_status,
                            f"Done! Path: {len(path)} steps · Cost: {cost:.2f}")
        else:
            self.root.after(0, self._set_status, "No path found.")

    def _update_tree(self, state):
        parent  = state["parent"]
        current = state["current"]
        if current in parent:
            self._tree.add_node(current)
            if parent[current] is not None:
                self._tree.add_edge(parent[current], current)
        for node in state["frontier"]:
            if node in parent and parent[node] is not None:
                self._tree.add_node(node)
                self._tree.add_edge(parent[node], node)

    # ══════════════════════════════════════════════════════════════════════
    #  DRAWING
    # ══════════════════════════════════════════════════════════════════════

    def _draw(self, state, path):
        if state is None:
            return
        self.ax_main.cla()
        self.ax_tree.cla()
        self.ax_metrics.cla()

        if self._graph_type == "grid":
            self._draw_grid(state, path)
        else:
            self._draw_city(state, path)

        self._draw_tree(state, path)
        self._draw_metrics(state, path)

        try:
            self.fig.tight_layout(rect=[0, 0, 1, 1])
        except Exception:
            pass
        self.canvas.draw()

    # ── Grid ─────────────────────────────────────────────────────────────

    def _draw_grid(self, state, path):
        env      = self._env
        ax       = self.ax_main
        visited  = state["visited"]
        frontier = set(state["frontier"])
        current  = state["current"]
        path_set = set(path)
        f_vals   = state.get("f", {})

        ax.set_facecolor(COLORS["empty"])
        ax.set_xlim(0, env.cols)
        ax.set_ylim(0, env.rows)
        ax.set_aspect("equal")

        # col labels (0-based) along bottom
        ax.set_xticks([c + 0.5 for c in range(env.cols)])
        ax.set_xticklabels([str(c) for c in range(env.cols)], fontsize=6, color=WM_MUTED)
        ax.xaxis.set_ticks_position("bottom")
        ax.set_xlabel("col", fontsize=6, color=WM_MUTED, labelpad=2)

        # row labels (0-based), top = row 0
        ax.set_yticks([env.rows - r - 0.5 for r in range(env.rows)])
        ax.set_yticklabels([str(r) for r in range(env.rows)], fontsize=6, color=WM_MUTED)
        ax.yaxis.set_ticks_position("left")
        ax.set_ylabel("row", fontsize=6, color=WM_MUTED, labelpad=2)

        ax.tick_params(length=0, pad=3)

        title = f"Grid  ·  {state.get('algo','?').upper()}  ·  {self.heuristic_var.get()}"
        if path:
            title += f"  ·  cost {_path_cost(env, path):.2f}"
        ax.set_title(title, color=WM_DARK, fontsize=10, pad=8)

        for r in range(env.rows):
            for c in range(env.cols):
                node = (r, c)
                if node in env.obstacles:          color = COLORS["obstacle"]
                elif node == env.start:            color = COLORS["start"]
                elif node == env.goal:             color = COLORS["goal"]
                elif node in path_set:             color = COLORS["path"]
                elif node == current:              color = COLORS["current"]
                elif node in frontier:             color = COLORS["frontier"]
                elif node in visited:              color = COLORS["visited"]
                else:                              color = COLORS["empty"]

                ax.add_patch(plt.Rectangle(
                    (c, env.rows - r - 1), 1, 1,
                    color=color, ec=COLORS["grid_line"], lw=0.4))

                if node == env.start:
                    ax.text(c+.5, env.rows-r-.5, "S", ha="center", va="center",
                            color="white", fontsize=7, fontweight="bold")
                elif node == env.goal:
                    ax.text(c+.5, env.rows-r-.5, "G", ha="center", va="center",
                            color="white", fontsize=7, fontweight="bold")
                elif node in frontier and node in f_vals:
                    ax.text(c+.5, env.rows-r-.5, f"{f_vals[node]:.1f}",
                            ha="center", va="center",
                            color=WM_DARK, fontsize=5)

        # Legend
        legend_items = [
            mpatches.Patch(color=COLORS["start"],    label="Start"),
            mpatches.Patch(color=COLORS["goal"],     label="Goal"),
            mpatches.Patch(color=COLORS["obstacle"], label="Obstacle"),
            mpatches.Patch(color=COLORS["visited"],  label="Visited"),
            mpatches.Patch(color=COLORS["frontier"], label="Frontier"),
            mpatches.Patch(color=COLORS["current"],  label="Current"),
            mpatches.Patch(color=COLORS["path"],     label="Path"),
        ]
        ax.legend(handles=legend_items, loc="upper left",
                  bbox_to_anchor=(1.01, 1), fontsize=7,
                  facecolor=WM_CARD, framealpha=0.95,
                  edgecolor=WM_BORDER)

        self._draw_queue_widget(ax, state)

    # ── City ─────────────────────────────────────────────────────────────

    def _draw_city(self, state, path):
        env      = self._env
        ax       = self.ax_main
        visited  = state["visited"]
        frontier = set(state["frontier"])
        current  = state["current"]
        path_set = set(zip(path, path[1:])) if len(path) > 1 else set()

        ax.set_facecolor("#faf6f0")
        ax.set_title(
            f"City Graph  ·  {state.get('algo','?').upper()}"
            + (f"  ·  cost {_path_cost(env, path):.2f}" if path else ""),
            color=WM_DARK, fontsize=10, pad=8)

        G = nx.Graph()
        for node in env.adjacency:
            G.add_node(node)
        for u, neighbors in env.adjacency.items():
            for entry in neighbors:
                v, w = entry if (isinstance(entry, tuple)
                                 and isinstance(entry[1], (int, float))) \
                             else (entry, 1.0)
                if not G.has_edge(u, v):
                    G.add_edge(u, v, weight=w)

        pos = self._city_pos or nx.spring_layout(G, seed=42)

        node_colors = []
        for node in G.nodes:
            if node == env.start:       node_colors.append(WM_PLUM)
            elif node == env.goal:      node_colors.append(WM_TERRA)
            elif node == current:       node_colors.append(WM_TERRA)
            elif node in visited:       node_colors.append(WM_SAGE)
            elif node in frontier:      node_colors.append(WM_CREAM)
            else:                       node_colors.append("#e8e2d8")

        edge_colors, edge_widths = [], []
        for u, v in G.edges:
            if (u, v) in path_set or (v, u) in path_set:
                edge_colors.append(WM_TERRA); edge_widths.append(3.0)
            else:
                edge_colors.append(WM_BORDER); edge_widths.append(0.7)

        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                               node_size=120, alpha=0.95)
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color=edge_colors,
                               width=edge_widths, alpha=0.6)
        labels = {n: n.replace("_", " ") for n in G.nodes}
        nx.draw_networkx_labels(G, pos, labels, ax=ax,
                                font_size=5.5, font_color=WM_DARK)
        if pos:
            xs = [x for x, y in pos.values()]
            ys = [y for x, y in pos.values()]
            pad = 0.15
            ax.set_xlim(min(xs) - pad, max(xs) + pad)
            ax.set_ylim(min(ys) - pad, max(ys) + pad)
        ax.axis("off")

        # Legend
        legend_items = [
            mpatches.Patch(color=WM_PLUM,    label="Start"),
            mpatches.Patch(color=WM_TERRA,   label="Goal / Current"),
            mpatches.Patch(color=WM_SAGE,    label="Visited"),
            mpatches.Patch(color=WM_CREAM,   label="Frontier"),
            mpatches.Patch(color="#e8e2d8",  label="Unvisited"),
        ]
        ax.legend(handles=legend_items, loc="upper left",
                  bbox_to_anchor=(1.01, 1), fontsize=7,
                  facecolor=WM_CARD, framealpha=0.95,
                  edgecolor=WM_BORDER)

        self._draw_queue_widget(ax, state)

    def _draw_tree(self, state, path):
        ax       = self.ax_tree
        tree     = self._tree
        current  = state["current"]
        path_set = set(zip(path, path[1:])) if len(path) > 1 else set()

        ax.set_facecolor(WM_BG)
        ax.set_title("Search Tree", color=WM_DARK, fontsize=10, pad=8)
        ax.axis("off")

        if not tree.nodes:
            return

        pos = self._tree_layout(tree, self._env.start)

        # Guard: only draw nodes that have a position
        valid_nodes = [n for n in tree.nodes if n in pos]
        valid_edges = [(u, v) for u, v in tree.edges if u in pos and v in pos]

        node_colors = []
        for node in valid_nodes:
            if node == self._env.start:     node_colors.append(WM_PLUM)
            elif node == self._env.goal:    node_colors.append(WM_TERRA)
            elif node == current:           node_colors.append(WM_TERRA)
            elif node in state["visited"]:  node_colors.append(WM_SAGE)
            else:                           node_colors.append(WM_CREAM)

        edge_colors = [
            WM_TERRA if ((u, v) in path_set or (v, u) in path_set) else "#b0a090"
            for u, v in valid_edges
        ]

        if self._graph_type == "city":
            labels = {n: n.replace("_", " ")[:7] for n in valid_nodes}
        else:
            labels = {n: f"{n[0]},{n[1]}" for n in valid_nodes}

        sub = tree.subgraph(valid_nodes)
        nx.draw(sub, pos, ax=ax,
                node_color=node_colors, edge_color=edge_colors,
                node_size=120, font_size=4.5, font_color=WM_DARK,
                arrows=True, arrowsize=6, width=1.0, labels=labels)

        # g/h/f annotation on current node
        g_vals = state.get("g", {})
        h_vals = state.get("h", {})
        f_vals = state.get("f", {})
        if current in pos:
            x, y = pos[current]
            g = g_vals.get(current, 0)
            h = h_vals.get(current, 0)
            f = f_vals.get(current, 0)
            ax.annotate(f"g={g:.1f}  h={h:.1f}  f={f:.1f}",
                        xy=(x, y), xytext=(x + 0.04, y + 0.06),
                        fontsize=5, color=WM_TERRA,
                        bbox=dict(facecolor=WM_CARD, alpha=0.85,
                                  boxstyle="round,pad=0.3",
                                  edgecolor=WM_BORDER))

    def _tree_layout(self, tree, root):
        from collections import deque
        pos, visited, levels = {}, set(), {}
        queue = deque([(root, 0)])
        while queue:
            node, depth = queue.popleft()
            if node in visited: continue
            visited.add(node)
            levels.setdefault(depth, []).append(node)
            for child in tree.successors(node):
                if child not in visited:
                    queue.append((child, depth + 1))
        max_d = max(levels) if levels else 0
        for depth, nodes in levels.items():
            for i, node in enumerate(nodes):
                pos[node] = ((i + 1) / (len(nodes) + 1),
                             1.0 - depth / (max_d + 1))
        return pos

    # ── Queue widget (inset under legend in main ax) ─────────────────────

    def _draw_queue_widget(self, ax, state):
        frontier_list = state["frontier"]
        f_vals_q      = state.get("f", {})
        algo          = state.get("algo", "")
        is_grid       = self._graph_type == "grid"

        def fmt_node(node):
            if is_grid and isinstance(node, tuple) and len(node) == 2:
                return f"({node[0]}, {node[1]})"
            return str(node)[:13]

        lines = [f"QUEUE  (size: {len(frontier_list)})"]
        lines.append("─" * 22)

        if algo == "iddfs":
            g_vals_q = state.get("g", {})
            current  = state["current"]
            lines.append("No frontier — IDDFS")
            lines.append("uses recursion")
            lines.append(f"g(cur)={g_vals_q.get(current,0):.2f}")
        elif frontier_list:
            if f_vals_q:
                display = sorted([(f_vals_q.get(n, 0), n) for n in frontier_list])
            else:
                display = [(0, n) for n in frontier_list]
            for fv, node in display:
                ns = fmt_node(node)
                fstr = f"  f={fv:.1f}" if f_vals_q else ""
                lines.append(f"{ns}{fstr}")
        else:
            lines.append("(empty)")

        ax.text(1.01, 0.44, "\n".join(lines),
                transform=ax.transAxes,
                fontsize=6, family="monospace",
                va="top", ha="left",
                color=WM_CREAM,
                bbox=dict(boxstyle="round,pad=0.5",
                          facecolor=WM_PLUM, alpha=0.92,
                          edgecolor=WM_PLUM),
                clip_on=False)

    # ── Metrics bar ───────────────────────────────────────────────────────

    def _draw_metrics(self, state, path):
        ax = self.ax_metrics
        ax.set_facecolor(WM_PANEL)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        g_vals  = state.get("g", {})
        h_vals  = state.get("h", {})
        f_vals  = state.get("f", {})
        current = state["current"]
        visited = state["visited"]
        frontier = state["frontier"]

        g_cur = g_vals.get(current, 0)
        h_cur = h_vals.get(current, 0)
        f_cur = f_vals.get(current, 0)
        pc    = f"{_path_cost(self._env, path):.3f}" if path else "—"
        depth = str(len(path) - 1) if path else "—"
        found = state.get("found", False)

        # ── Metric tiles ─────────────────────────────────────────────────
        tiles = [
            ("ALGORITHM",   state.get("algo", "?").upper()),
            ("EXPANDED",    str(len(visited))),
            ("FRONTIER",    str(len(frontier))),
            ("g(n)",        f"{g_cur:.2f}"),
            ("h(n)",        f"{h_cur:.2f}"),
            ("f(n)",        f"{f_cur:.2f}"),
            ("PATH COST",   pc),
            ("DEPTH",       depth),
            ("STATUS",      "FOUND" if found else "searching…"),
        ]

        n     = len(tiles)
        # Full width for metric tiles now that queue moved to main ax
        total_w = 1.0
        tile_w  = total_w / n
        pad     = 0.004

        for i, (label, value) in enumerate(tiles):
            x0 = i * tile_w + pad
            x1 = (i + 1) * tile_w - pad

            is_found  = (label == "STATUS" and found)
            box_color = "#d4edda" if is_found else WM_CARD
            txt_color = "#1a7a3a" if is_found else WM_DARK
            lbl_color = WM_MUTED

            # Box
            ax.add_patch(__import__("matplotlib").patches.FancyBboxPatch(
                (x0, 0.08), x1 - x0, 0.84,
                boxstyle="round,pad=0.01",
                facecolor=box_color, edgecolor=WM_BORDER, linewidth=0.7,
                transform=ax.transAxes, clip_on=False))

            # Label (top)
            ax.text((x0 + x1) / 2, 0.78, label,
                    transform=ax.transAxes,
                    ha="center", va="center",
                    fontsize=6, color=lbl_color,
                    fontfamily="DejaVu Sans")

            # Value (bottom, big)
            ax.text((x0 + x1) / 2, 0.38, value,
                    transform=ax.transAxes,
                    ha="center", va="center",
                    fontsize=9, color=txt_color,
                    fontweight="bold", fontfamily="monospace")


    # ── Frontier widget ───────────────────────────────────────────────────

    def _draw_frontier_widget(self, ax, state):
        frontier = state["frontier"]
        f_vals   = state.get("f", {})
        if not frontier:
            return
        lines = [f"{'Queue':^24}", f"size: {len(frontier)}",
                 "─" * 24]
        display = (sorted([(f_vals.get(n, 0), n) for n in frontier])[:6]
                   if f_vals else [(0, n) for n in frontier[:6]])
        for fv, node in display:
            ns = f"{node}" if isinstance(node, tuple) else str(node)[:12]
            lines.append(f"  {ns:<16}" + (f" {fv:.1f}" if f_vals else ""))
        if len(frontier) > 6:
            lines.append(f"  +{len(frontier)-6} more…")

        ax.text(0.01, 0.01, "\n".join(lines),
                transform=ax.transAxes,
                fontsize=5, family="monospace",
                va="bottom", ha="left",
                color=WM_CARD,
                bbox=dict(boxstyle="round,pad=0.4",
                          facecolor=WM_PLUM, alpha=0.82,
                          edgecolor=WM_PLUM))

    # ══════════════════════════════════════════════════════════════════════
    #  HOVER / CLICK TOOLTIPS
    # ══════════════════════════════════════════════════════════════════════

    def _on_mouse_move(self, event):
        """Show tooltip on hover — only after search finishes."""
        if self._running or self._final_state is None:
            return
        if self._tooltip_pinned:
            return
        if event.inaxes != self.ax_main:
            self._clear_tooltip()
            return
        self._show_tooltip(event.xdata, event.ydata, pin=False)

    def _on_mouse_click(self, event):
        """Pin/unpin tooltip on click — only after search finishes."""
        if self._running or self._final_state is None:
            return
        if event.inaxes != self.ax_main:
            return
        if self._tooltip_pinned:
            # Second click unpins
            self._tooltip_pinned = False
            self._clear_tooltip()
        else:
            self._show_tooltip(event.xdata, event.ydata, pin=True)

    def _show_tooltip(self, x, y, pin=False):
        """Find the node nearest to (x, y) and show its g/h/f/parent info."""
        if x is None or y is None:
            return

        state = self._final_state
        g_vals  = state.get("g", {})
        h_vals  = state.get("h", {})
        f_vals  = state.get("f", {})
        parent  = state.get("parent", {})

        node = None

        if self._graph_type == "grid":
            env  = self._env
            col  = int(x)
            row  = env.rows - 1 - int(y)
            node = (row, col)
            if node in env.obstacles or not (0 <= row < env.rows and 0 <= col < env.cols):
                self._clear_tooltip()
                return
            # Tooltip anchor at cell center
            tx = col + 0.5
            ty = env.rows - row - 0.5

        else:
            # City graph — find nearest node by Euclidean distance to pos
            pos = self._city_pos
            if not pos:
                return
            best_dist = float("inf")
            for name, (px, py) in pos.items():
                d = (px - x)**2 + (py - y)**2
                if d < best_dist:
                    best_dist = d
                    node = name
            # Only show if close enough (within ~8% of axes range)
            xlim = self.ax_main.get_xlim()
            threshold = ((xlim[1] - xlim[0]) * 0.08) ** 2
            if best_dist > threshold:
                self._clear_tooltip()
                return
            tx, ty = pos[node]

        # ── Build tooltip text ────────────────────────────────────────────
        g = g_vals.get(node, "—")
        h = h_vals.get(node, "—")
        f = f_vals.get(node, "—")
        par = parent.get(node, None)

        if isinstance(g, float): g = f"{g:.2f}"
        if isinstance(h, float): h = f"{h:.2f}"
        if isinstance(f, float): f = f"{f:.2f}"

        if self._graph_type == "grid":
            node_label = f"({node[0]}, {node[1]})"
            par_label  = f"({par[0]}, {par[1]})" if isinstance(par, tuple) else "None"
        else:
            node_label = str(node).replace("_", " ")
            par_label  = str(par).replace("_", " ") if par else "None"

        pin_note = " [click to unpin]" if pin else " [click to pin]"
        lines = [
            f"Node:   {node_label}",
            f"g(n):   {g}",
            f"h(n):   {h}",
            f"f(n):   {f}",
            f"parent: {par_label}",
            pin_note,
        ]
        text = "\n".join(lines)

        # ── Draw annotation ───────────────────────────────────────────────
        self._clear_tooltip(redraw=False)

        self._tooltip_annotation = self.ax_main.annotate(
            text,
            xy=(tx, ty),
            xytext=(18, 18), textcoords="offset points",
            fontsize=6.5, family="monospace",
            color=WM_CREAM,
            bbox=dict(boxstyle="round,pad=0.5",
                      facecolor=WM_PLUM, alpha=0.95,
                      edgecolor=WM_BORDER),
            arrowprops=dict(arrowstyle="->", color=WM_MUTED, lw=0.8),
            zorder=10,
        )

        self._tooltip_pinned = pin
        self.canvas.draw_idle()

    def _clear_tooltip(self, redraw=True):
        if self._tooltip_annotation is not None:
            try:
                self._tooltip_annotation.remove()
            except Exception:
                pass
            self._tooltip_annotation = None
            if redraw:
                self.canvas.draw_idle()



    def _compute_city_layout(self, env):
        layout = self.layout_var.get()

        G = nx.Graph()
        for node in env.adjacency:
            G.add_node(node)
        for u, neighbors in env.adjacency.items():
            for entry in neighbors:
                v, w = entry if (isinstance(entry, tuple)
                                 and isinstance(entry[1], (int, float))) \
                             else (entry, 1.0)
                if not G.has_edge(u, v):
                    G.add_edge(u, v, weight=w)

        if layout == "geographic":
            coords = env.coords
            lats = [v[0] for v in coords.values()]
            lons = [v[1] for v in coords.values()]
            lat_min, lat_max = min(lats), max(lats)
            lon_min, lon_max = min(lons), max(lons)
            lat_r = (lat_max - lat_min) or 1
            lon_r = (lon_max - lon_min) or 1
            SCALE, MARGIN = 3.0, 0.2
            return {
                nid: (MARGIN + SCALE * (lon - lon_min) / lon_r,
                      MARGIN + SCALE * (lat - lat_min) / lat_r)
                for nid, (lat, lon) in coords.items()
            }
        else:  # spring
            return nx.spring_layout(G, seed=42, k=1.2)


# ══════════════════════════════════════════════════════════════════════════════
#  Utility
# ══════════════════════════════════════════════════════════════════════════════

def _path_cost(env, path):
    cost = 0.0
    for i in range(len(path) - 1):
        if hasattr(env, "edge_weight"):
            cost += env.edge_weight(path[i], path[i+1])
        else:
            dr = abs(path[i][0] - path[i+1][0])
            dc = abs(path[i][1] - path[i+1][1])
            cost += 1.414 if (dr + dc == 2) else 1.0
    return round(cost, 3)


def _darken(hex_color, amount=0.12):
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i+2], 16) for i in (0, 2, 4))
    return "#{:02x}{:02x}{:02x}".format(
        max(0, int(r * (1 - amount))),
        max(0, int(g * (1 - amount))),
        max(0, int(b * (1 - amount))),
    )