"""
launcher.py  — Two-page Tkinter launcher for AI Search Visualizer
Page 1: Graph mode + environment settings
Page 2: Algorithm, heuristic, start/goal, speed → Launch

# Claude Prompt:
# Build a two-page Tkinter launcher for an AI search visualizer. Page 1 should let
# the user select graph mode (grid or city) and configure grid settings (rows, cols,
# obstacle density, connectivity). Page 2 should let the user pick an algorithm,
# heuristic, start/goal cities, and animation speed. Use a dark color palette and
# return the selected config as a dict when the user clicks Launch.
"""

import tkinter as tk
from tkinter import ttk, messagebox

# ── Palette ───────────────────────────────────────────────────────────────────
BG         = "#0d0f1a"
PANEL      = "#131629"
CARD       = "#1a1f35"
BORDER     = "#252d4a"
ACCENT     = "#4f8ef7"
ACCENT2    = "#a78bfa"
SUCCESS    = "#34d399"
TEXT       = "#e8ecf5"
TEXT_DIM   = "#6b7592"
TEXT_MUTED = "#3d4560"
WHITE      = "#ffffff"

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
    ("A*",     "astar",  "Optimal + complete\ng(n)+h(n) guided"),
    ("Greedy", "greedy", "Fast, not optimal\nExpands by h(n)"),
    ("BFS",    "bfs",    "Optimal unit cost\nLevel-by-level"),
    ("DFS",    "dfs",    "Low memory\nDives deep first"),
    ("IDDFS",  "iddfs",  "BFS optimal\nDFS memory"),
]


def _lighten(hex_color, amount=0.15):
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i+2], 16) for i in (0, 2, 4))
    return "#{:02x}{:02x}{:02x}".format(
        min(255, int(r + (255-r)*amount)),
        min(255, int(g + (255-g)*amount)),
        min(255, int(b + (255-b)*amount)),
    )


def _style_combo(cb):
    s = ttk.Style()
    s.theme_use("clam")
    s.configure("D.TCombobox",
        fieldbackground=CARD, background=BORDER,
        foreground=TEXT, selectbackground=ACCENT,
        selectforeground=WHITE, bordercolor=BORDER,
        lightcolor=BORDER, darkcolor=BORDER, arrowcolor=ACCENT)
    s.map("D.TCombobox", fieldbackground=[("readonly", CARD)],
          foreground=[("readonly", TEXT)])
    cb.configure(style="D.TCombobox", font=("Courier New", 10))


class LauncherGUI:
    def __init__(self):
        self.result = None
        self.root   = tk.Tk()
        self.root.title("AI Search Visualizer")
        self.root.configure(bg=BG)
        self.root.resizable(False, False)

        W, H = 660, 560
        sw, sh = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        self.root.geometry(f"{W}x{H}+{(sw-W)//2}+{(sh-H)//2}")

        # ── Shared state ──────────────────────────────────────────────────
        self.mode_var      = tk.StringVar(value="grid")
        self.algo_var      = tk.StringVar(value="astar")
        self.heuristic_var = tk.StringVar(value="manhattan")
        self.rows_var      = tk.IntVar(value=10)
        self.cols_var      = tk.IntVar(value=10)
        self.obs_var       = tk.DoubleVar(value=0.25)
        self.conn_var      = tk.IntVar(value=4)
        self.speed_var     = tk.DoubleVar(value=0.15)
        self.start_var     = tk.StringVar(value="Wichita")
        self.goal_var      = tk.StringVar(value="Topeka")

        self._mode_tiles  = {}
        self._algo_tiles  = {}

        self.container = tk.Frame(self.root, bg=BG)
        self.container.pack(fill=tk.BOTH, expand=True)

        self.p1 = tk.Frame(self.container, bg=BG)
        self.p2 = tk.Frame(self.container, bg=BG)

        self._build_p1()
        self._build_p2()
        self._goto(1)

        self.root.mainloop()

    # ── Page navigation ───────────────────────────────────────────────────────

    def _goto(self, n):
        self.p1.place_forget()
        self.p2.place_forget()
        if n == 1:
            self.p1.place(x=0, y=0, relwidth=1, relheight=1)
        else:
            self._sync_p2()
            self.p2.place(x=0, y=0, relwidth=1, relheight=1)

    # ── Shared builders ───────────────────────────────────────────────────────

    def _header(self, parent, step, title, sub):
        f = tk.Frame(parent, bg=BG)
        f.pack(fill=tk.X, padx=36, pady=(28, 0))
        tk.Label(f, text=f"STEP {step} / 2",
                 font=("Courier New", 8, "bold"), bg=BG, fg=ACCENT).pack(anchor="w")
        tk.Label(f, text=title, font=("Georgia", 21, "bold"),
                 bg=BG, fg=TEXT).pack(anchor="w", pady=(3, 0))
        tk.Label(f, text=sub, font=("Courier New", 9),
                 bg=BG, fg=TEXT_DIM).pack(anchor="w", pady=(1, 0))
        sep = tk.Frame(f, bg=ACCENT, height=2)
        sep.pack(fill=tk.X, pady=(10, 0))

    def _section_label(self, parent, text):
        tk.Label(parent, text=text.upper(),
                 font=("Courier New", 8, "bold"),
                 bg=BG, fg=TEXT_MUTED).pack(anchor="w", pady=(0, 4))

    def _card_frame(self, parent):
        f = tk.Frame(parent, bg=CARD,
                     highlightbackground=BORDER, highlightthickness=1)
        f.pack(fill=tk.X)
        return f

    def _nav_btn(self, parent, text, cmd, color=ACCENT, fg=WHITE, side=tk.LEFT):
        b = tk.Button(parent, text=text, command=cmd,
                      bg=color, fg=fg, activebackground=_lighten(color),
                      activeforeground=fg,
                      font=("Courier New", 10, "bold"),
                      relief=tk.FLAT, bd=0,
                      padx=22, pady=10, cursor="hand2")
        b.pack(side=side)
        b.bind("<Enter>", lambda e: b.configure(bg=_lighten(color)))
        b.bind("<Leave>", lambda e: b.configure(bg=color))
        return b

    # ── PAGE 1 ────────────────────────────────────────────────────────────────

    def _build_p1(self):
        p = self.p1
        self._header(p, 1, "Environment Setup",
                     "Choose graph type and configure the world")

        # Mode tiles
        outer = tk.Frame(p, bg=BG)
        outer.pack(fill=tk.X, padx=36, pady=(20, 14))
        self._section_label(outer, "Graph Mode")
        card = self._card_frame(outer)

        row = tk.Frame(card, bg=CARD)
        row.pack(fill=tk.X, padx=14, pady=12)
        row.columnconfigure(0, weight=1)
        row.columnconfigure(1, weight=1)

        for col, (icon, label, val, desc) in enumerate([
            ("⊞", "Grid World",          "grid",
             "Random maze  ·  obstacle density\n4- or 8-way connectivity  ·  seeded"),
            ("◉", "Kansas City Graph",   "city",
             "45 real Kansas cities\nconnected by road distances"),
        ]):
            self._mode_tile(row, col, icon, label, val, desc)

        # Grid settings
        self.grid_section = tk.Frame(p, bg=BG)
        self.grid_section.pack(fill=tk.X, padx=36, pady=(0, 12))
        self._section_label(self.grid_section, "Grid Settings")
        gc = self._card_frame(self.grid_section)
        gi = tk.Frame(gc, bg=CARD)
        gi.pack(fill=tk.X, padx=16, pady=12)

        for col, (lbl, var, lo, hi, inc) in enumerate([
            ("Rows",       self.rows_var, 5,    30,   1),
            ("Cols",       self.cols_var, 5,    30,   1),
            ("Obstacle %", self.obs_var,  0.20, 0.30, 0.01),
        ]):
            f = tk.Frame(gi, bg=CARD)
            f.grid(row=0, column=col, padx=14, sticky="w")
            gi.columnconfigure(col, weight=1)
            tk.Label(f, text=lbl, font=("Courier New", 8),
                     bg=CARD, fg=TEXT_DIM).pack(anchor="w")
            tk.Spinbox(f, from_=lo, to=hi, increment=inc, textvariable=var,
                       width=7, font=("Courier New", 11, "bold"),
                       bg=BORDER, fg=TEXT, insertbackground=TEXT,
                       relief=tk.FLAT, bd=3,
                       buttonbackground=BORDER).pack(pady=(2, 0))

        cf = tk.Frame(gi, bg=CARD)
        cf.grid(row=0, column=3, padx=14, sticky="w")
        tk.Label(cf, text="Connectivity", font=("Courier New", 8),
                 bg=CARD, fg=TEXT_DIM).pack(anchor="w")
        rbf = tk.Frame(cf, bg=CARD)
        rbf.pack(pady=(2, 0))
        for v, t in [(4, "4-way"), (8, "8-way")]:
            tk.Radiobutton(rbf, text=t, variable=self.conn_var, value=v,
                           bg=CARD, fg=TEXT, selectcolor=ACCENT,
                           activebackground=CARD,
                           font=("Courier New", 9),
                           indicatoron=0, relief=tk.FLAT,
                           padx=8, pady=4, cursor="hand2").pack(side=tk.LEFT, padx=2)

        tk.Label(gc, text="Start & Goal placed randomly — override on next page",
                 font=("Courier New", 7), bg=CARD, fg=TEXT_MUTED).pack(pady=(0, 7))

        # City note
        self.city_section = tk.Frame(p, bg=BG)
        tk.Label(self.city_section, text="CITY GRAPH",
                 font=("Courier New", 8, "bold"),
                 bg=BG, fg=TEXT_MUTED).pack(anchor="w", pady=(0, 4))
        cc = self._card_frame(self.city_section)
        tk.Label(cc,
                 text="45 Kansas cities · road distances · start & goal on next page",
                 font=("Courier New", 10), bg=CARD, fg=TEXT_DIM,
                 justify=tk.LEFT).pack(padx=16, pady=14, anchor="w")

        # Nav
        nav = tk.Frame(p, bg=BG)
        nav.pack(fill=tk.X, padx=36, pady=(12, 0))
        self._nav_btn(nav, "Next  →", lambda: self._goto(2),
                      color=ACCENT, side=tk.RIGHT)

        self._refresh_mode()

    def _mode_tile(self, parent, col, icon, label, val, desc):
        def select():
            self.mode_var.set(val)
            self._refresh_mode()

        tile = tk.Frame(parent, bg=CARD, cursor="hand2",
                        highlightthickness=2, highlightbackground=BORDER)
        tile.grid(row=0, column=col, padx=6, pady=4, sticky="nsew")

        for widget_cls, text, font, fg in [
            (tk.Label, icon,  ("Courier New", 24), ACCENT),
            (tk.Label, label, ("Courier New", 10, "bold"), TEXT),
            (tk.Label, desc,  ("Courier New", 8),  TEXT_DIM),
        ]:
            w = widget_cls(tile, text=text, font=font, bg=CARD, fg=fg,
                           justify=tk.CENTER)
            w.pack(pady=(10 if text == icon else 1, 10 if text == desc else 0),
                   padx=8)
            w.bind("<Button-1>", lambda e, s=select: s())
        tile.bind("<Button-1>", lambda e, s=select: s())

        self._mode_tiles[val] = tile

    def _refresh_mode(self):
        mode = self.mode_var.get()
        for val, tile in self._mode_tiles.items():
            sel = val == mode
            tile.configure(
                highlightbackground=ACCENT if sel else BORDER,
                highlightthickness=2,
                bg=PANEL if sel else CARD)
            for w in tile.winfo_children():
                w.configure(bg=PANEL if sel else CARD)

        if mode == "grid":
            self.city_section.pack_forget()
            self.grid_section.pack(fill=tk.X, padx=36, pady=(0, 12))
        else:
            self.grid_section.pack_forget()
            self.city_section.pack(fill=tk.X, padx=36, pady=(0, 12))

    # ── PAGE 2 ────────────────────────────────────────────────────────────────

    def _build_p2(self):
        p = self.p2
        self._header(p, 2, "Search Configuration",
                     "Pick your algorithm, nodes, and launch")

        # Algorithm tiles
        ao = tk.Frame(p, bg=BG)
        ao.pack(fill=tk.X, padx=36, pady=(20, 12))
        self._section_label(ao)
        ac = self._card_frame(ao)
        ai = tk.Frame(ac, bg=CARD)
        ai.pack(fill=tk.X, padx=10, pady=10)
        for col, (name, val, desc) in enumerate(ALGOS):
            ai.columnconfigure(col, weight=1)
            self._algo_tile(ai, col, name, val, desc)

        # Heuristic
        ho = tk.Frame(p, bg=BG)
        ho.pack(fill=tk.X, padx=36, pady=(0, 12))
        self._section_label(ho, "Heuristic")
        hc = self._card_frame(ho)
        hi = tk.Frame(hc, bg=CARD)
        hi.pack(fill=tk.X, padx=16, pady=10)
        self.h_combo = ttk.Combobox(hi, textvariable=self.heuristic_var,
                                    values=GRID_HEURISTICS, state="readonly", width=30)
        _style_combo(self.h_combo)
        self.h_combo.pack(side=tk.LEFT)

        # Start / Goal (city mode)
        self.node_outer = tk.Frame(p, bg=BG)
        self.node_outer.pack(fill=tk.X, padx=36, pady=(0, 12))
        self._section_label(self.node_outer, "Start & Goal")
        nc = self._card_frame(self.node_outer)
        ni = tk.Frame(nc, bg=CARD)
        ni.pack(fill=tk.X, padx=16, pady=10)

        for side_val, label, var, attr in [
            (tk.LEFT,  "Start City", self.start_var, "start_combo"),
            (tk.LEFT,  "Goal City",  self.goal_var,  "goal_combo"),
        ]:
            sf = tk.Frame(ni, bg=CARD)
            sf.pack(side=tk.LEFT, padx=(0, 20))
            tk.Label(sf, text=label, font=("Courier New", 8),
                     bg=CARD, fg=TEXT_DIM).pack(anchor="w")
            cb = ttk.Combobox(sf, textvariable=var,
                              values=CITIES, state="readonly", width=18)
            _style_combo(cb)
            cb.pack(pady=(2, 0))
            setattr(self, attr, cb)
            if label == "Start City":
                tk.Label(ni, text="→", font=("Georgia", 16),
                         bg=CARD, fg=ACCENT).pack(side=tk.LEFT, padx=8)

        self.node_note = tk.Label(nc, text="",
                                  font=("Courier New", 7), bg=CARD, fg=TEXT_MUTED)
        self.node_note.pack(pady=(0, 6))

        # Speed
        so = tk.Frame(p, bg=BG)
        so.pack(fill=tk.X, padx=36, pady=(0, 14))
        self._section_label(so, "Animation Speed")
        sc = self._card_frame(so)
        si = tk.Frame(sc, bg=CARD)
        si.pack(fill=tk.X, padx=16, pady=10)
        tk.Label(si, text="Fast", font=("Courier New", 8),
                 bg=CARD, fg=TEXT_DIM).pack(side=tk.LEFT)
        tk.Scale(si, variable=self.speed_var,
                 from_=0.02, to=1.0, resolution=0.01,
                 orient=tk.HORIZONTAL, length=320,
                 bg=CARD, fg=TEXT, troughcolor=BORDER,
                 activebackground=ACCENT, highlightthickness=0,
                 showvalue=True, font=("Courier New", 7),
                 sliderrelief=tk.FLAT).pack(side=tk.LEFT, padx=8)
        tk.Label(si, text="Slow", font=("Courier New", 8),
                 bg=CARD, fg=TEXT_DIM).pack(side=tk.LEFT)

        # Nav
        nav = tk.Frame(p, bg=BG)
        nav.pack(fill=tk.X, padx=36)
        self._nav_btn(nav, "←  Back", lambda: self._goto(1),
                      color=BORDER, fg=TEXT_DIM, side=tk.LEFT)
        self._nav_btn(nav, "▶  Launch Search", self._on_run,
                      color=SUCCESS, fg="#0d0f1a", side=tk.RIGHT)

    def _section_label(self, parent, text="Algorithm"):
        tk.Label(parent, text=text.upper(),
                 font=("Courier New", 8, "bold"),
                 bg=BG, fg=TEXT_MUTED).pack(anchor="w", pady=(0, 4))

    def _algo_tile(self, parent, col, name, val, desc):
        def select():
            self.algo_var.set(val)
            self._refresh_algos()

        tile = tk.Frame(parent, bg=CARD, cursor="hand2",
                        highlightthickness=1, highlightbackground=BORDER)
        tile.grid(row=0, column=col, padx=4, sticky="nsew")
        tile.bind("<Button-1>", lambda e: select())

        nl = tk.Label(tile, text=name, font=("Courier New", 10, "bold"),
                      bg=CARD, fg=TEXT, cursor="hand2")
        nl.pack(pady=(10, 2), padx=6)
        nl.bind("<Button-1>", lambda e: select())

        dl = tk.Label(tile, text=desc, font=("Courier New", 7),
                      bg=CARD, fg=TEXT_DIM, justify=tk.CENTER, wraplength=100)
        dl.pack(pady=(0, 10), padx=4)
        dl.bind("<Button-1>", lambda e: select())

        self._algo_tiles[val] = (tile, nl, dl)

    def _refresh_algos(self):
        sel = self.algo_var.get()
        for val, (tile, nl, dl) in self._algo_tiles.items():
            chosen = val == sel
            bg = PANEL if chosen else CARD
            tile.configure(highlightbackground=ACCENT2 if chosen else BORDER,
                           highlightthickness=2 if chosen else 1, bg=bg)
            nl.configure(bg=bg, fg=ACCENT2 if chosen else TEXT)
            dl.configure(bg=bg)

    def _sync_p2(self):
        mode = self.mode_var.get()
        if mode == "grid":
            self.h_combo.configure(values=GRID_HEURISTICS)
            if self.heuristic_var.get() not in GRID_HEURISTICS:
                self.heuristic_var.set("manhattan")
            self.start_combo.configure(state="disabled")
            self.goal_combo.configure(state="disabled")
            self.node_note.configure(
                text="Grid mode: start & goal are randomly placed")
        else:
            self.h_combo.configure(values=CITY_HEURISTICS)
            if self.heuristic_var.get() not in CITY_HEURISTICS:
                self.heuristic_var.set("euclidean_coords")
            self.start_combo.configure(state="readonly")
            self.goal_combo.configure(state="readonly")
            self.node_note.configure(text="")
        self._refresh_algos()

    def _on_run(self):
        mode  = self.mode_var.get()
        start = self.start_var.get()
        goal  = self.goal_var.get()

        if mode == "city" and start == goal:
            messagebox.showerror("Invalid", "Start and goal must be different cities.")
            return

        self.result = {
            "graph"     : mode,
            "method"    : self.algo_var.get(),
            "heuristic" : self.heuristic_var.get(),
            "speed"     : round(self.speed_var.get(), 3),
            "rows"      : self.rows_var.get(),
            "cols"      : self.cols_var.get(),
            "obstacles" : round(self.obs_var.get(), 2),
            "conn"      : self.conn_var.get(),
            "start_city": start,
            "goal_city" : goal,
        }
        self.root.destroy()


def launch() -> dict | None:
    gui = LauncherGUI()
    return gui.result