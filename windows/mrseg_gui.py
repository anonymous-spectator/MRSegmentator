# Copyright 2024-2026 Hartmut Häntze
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0

"""Simplified graphical front-end for the Windows executable.

``mrseg_entry.py`` opens this window when ``mrsegmentator.exe`` is started
with no arguments at all -- the situation a double-click in Explorer always
produces. Started with real arguments (from a terminal, a script, ...) the
executable behaves exactly like the pip installation instead; this module is
never imported on that path.

The GUI itself never touches inference code. Each run is simply the same CLI
the pip installation exposes, invoked as a subprocess with the model and
options the user picked -- fast mode and split level 1 are pre-selected
defaults, not hard-coded:

    mrsegmentator.exe --input <file-or-folder> --outdir <dir> --fast
                       --split_level {0,1,2} [--body_comp]

Running it out-of-process rather than importing ``mrsegmentator.main``
directly keeps this file lightweight (no torch/nnU-Net import here, so the
window opens instantly) and lets the log pane simply show whatever the CLI
itself prints, tqdm bars included.

The header at the top is purely informational: the project name, a note that
this simplified GUI runs a CPU-friendly light mode, and clickable links to
the codebase and the two papers.
"""

import os
import queue
import re
import subprocess
import sys
import threading
import tkinter as tk
import webbrowser
from pathlib import Path
from tkinter import filedialog, font as tkfont, messagebox, ttk
from tkinter.scrolledtext import ScrolledText
from typing import Any, List, Optional, Tuple

WINDOWS_DIR = Path(__file__).resolve().parent
ENTRY_SCRIPT = WINDOWS_DIR / "mrseg_entry.py"

SUPPORTED_EXTENSIONS = (".nii", ".nii.gz", ".mha", ".nrrd")
PERCENT_RE = re.compile(r"(\d{1,3})\s*%")

# Cap the log pane so a very long, very chatty run cannot grow it forever.
MAX_LOG_LINES = 4000

# Shown in the header; kept small and non-technical since this is the
# simplified GUI, not the CLI.
PRODUCT_NAME = "MRSegmentator"
SUBTITLE = "Multi-Modality Segmentation of 40+10 Classes"
LIGHT_MODE_NOTE = "⚡ CPU-friendly light mode: single fold, fast settings for quick results."

# (label, url, fill color, hover color) -- one shields.io-style badge each.
BADGES: List[Tuple[str, str, str, str]] = [
    ("Codebase", "https://github.com/hhaentze/MRSegmentator", "#24292f", "#3a4149"),
    ("Main Paper", "https://pubs.rsna.org/doi/abs/10.1148/ryai.240777", "#1a73e8", "#0b57c7"),
    (
        "Bodycomp Paper",
        "https://www.nature.com/articles/s43856-026-01888-w",
        "#0f9d76",
        "#0c7d5e",
    ),
]

HEADER_BG = "#eef3fc"
TITLE_COLOR = "#1f3a5f"
ACCENT_COLOR = "#b4530f"
MUTED_COLOR = "#666666"
BADGE_TEXT_COLOR = "#ffffff"


def _is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False)) or "__compiled__" in globals()


def _apply_window_icon(root: tk.Tk) -> None:
    """Set the window/taskbar icon to the one built with ``--icon``, if any.

    Tk does not inherit the hosting .exe's own PE icon resource for its
    window decorations (that's a separate, Windows-only concept), so
    build_windows_exe.py also bundles the same normalized icon as a plain
    data file under the fixed name ``icon.ico`` -- all this needs is to find
    that file and hand it to Tk. A no-op outside a frozen build (there is
    nothing bundled to find) or if anything about this goes wrong -- a
    missing custom icon is not worth failing the GUI over.
    """
    if not _is_frozen():
        return
    try:
        import frozen_support

        icon_path = frozen_support.find_data_file("icon.ico")
        if icon_path is not None:
            root.iconbitmap(str(icon_path))
    except Exception:
        pass


def _cli_command(args: List[str]) -> List[str]:
    """Build the command line that re-invokes MRSegmentator's own CLI.

    In the built .exe, ``sys.executable`` *is* mrsegmentator.exe, so passing
    real arguments to it takes the normal (unmodified) CLI path -- including
    ``--multiprocessing-fork`` re-entry for worker processes, which is why
    the GUI must launch a fresh process per run rather than call
    ``mrsegmentator.main.main()`` in-process.

    Run from a plain interpreter (e.g. while developing this file outside a
    build), fall back to running ``mrseg_entry.py`` with that interpreter.
    """
    if _is_frozen():
        return [sys.executable, *args]
    return [sys.executable, str(ENTRY_SCRIPT), *args]


def _display_name(path: str) -> str:
    p = Path(path)
    return p.name if p.name else path


class _Badge(tk.Canvas):
    """A small clickable, rounded, colored link -- shields.io-badge style.

    Tk/ttk have no built-in rounded button, so this draws one: a rounded-rect
    polygon plus centered text on a Canvas sized to fit that text, redrawn in
    a slightly darker shade on hover. ``parent_bg`` must match the surrounding
    widget's actual background so the canvas's own (necessarily rectangular)
    corners are invisible against it.
    """

    _PAD_X = 14
    _PAD_Y = 6

    def __init__(
        self, parent: tk.Widget, text: str, url: str, fill: str, hover_fill: str, parent_bg: str
    ) -> None:
        self._font = tkfont.Font(family="Segoe UI", size=9, weight="bold")
        width = self._font.measure(text) + 2 * self._PAD_X
        height = self._font.metrics("linespace") + 2 * self._PAD_Y
        super().__init__(
            parent,
            width=width,
            height=height,
            highlightthickness=0,
            bd=0,
            bg=parent_bg,
            cursor="hand2",
        )
        self._text = text
        self._url = url
        self._fill = fill
        self._hover_fill = hover_fill
        self._width = width
        self._height = height
        self._paint(fill)
        self.bind("<Button-1>", lambda _event: webbrowser.open(self._url))
        self.bind("<Enter>", lambda _event: self._paint(self._hover_fill))
        self.bind("<Leave>", lambda _event: self._paint(self._fill))

    def _paint(self, color: str) -> None:
        self.delete("all")
        radius = self._height / 2
        self._rounded_rect(0, 0, self._width, self._height, radius, fill=color, outline=color)
        self.create_text(
            self._width / 2,
            self._height / 2,
            text=self._text,
            fill=BADGE_TEXT_COLOR,
            font=self._font,
        )

    def _rounded_rect(self, x1: float, y1: float, x2: float, y2: float, r: float, **opts: Any) -> int:
        points = [
            x1 + r, y1,
            x2 - r, y1,
            x2, y1,
            x2, y1 + r,
            x2, y2 - r,
            x2, y2,
            x2 - r, y2,
            x1 + r, y2,
            x1, y2,
            x1, y2 - r,
            x1, y1 + r,
            x1, y1,
        ]
        return self.create_polygon(points, smooth=True, **opts)


class _Job:
    """One queued input path (a file, a NIfTI directory, or a DICOM directory)."""

    def __init__(self, path: str):
        self.path = path
        self.name = _display_name(path)


class MRSegGUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title(PRODUCT_NAME)
        self.root.minsize(640, 640)

        self._jobs: List[_Job] = []
        self._queue: "queue.Queue[Tuple[str, object]]" = queue.Queue()
        self._proc: Optional[subprocess.Popen] = None
        self._worker: Optional[threading.Thread] = None
        self._cancel_requested = False
        self._running = False

        self.model_var = tk.StringVar(value="base")
        self.split_level_var = tk.StringVar(value="1")
        self.fast_var = tk.BooleanVar(value=True)
        self.outdir_var = tk.StringVar(value="")
        self.overall_status_var = tk.StringVar(value="Idle")
        self.current_status_var = tk.StringVar(value="")

        self._build_layout()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(100, self._poll_queue)

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------
    def _build_layout(self) -> None:
        pad = {"padx": 10, "pady": 6}
        root = self.root
        root.columnconfigure(0, weight=1)

        # Frame/Label backgrounds are honored on every built-in ttk theme
        # (unlike Button/Entry, which mostly ignore custom colors on Windows'
        # native theme), so a couple of named styles are enough to give the
        # header its own tinted panel without touching the rest of the app.
        style = ttk.Style(root)
        style.configure("Header.TFrame", background=HEADER_BG)
        style.configure("Header.TLabel", background=HEADER_BG)
        style.configure(
            "Title.Header.TLabel", background=HEADER_BG, foreground=TITLE_COLOR
        )
        style.configure(
            "Subtitle.Header.TLabel", background=HEADER_BG, foreground=MUTED_COLOR
        )
        style.configure(
            "Note.Header.TLabel", background=HEADER_BG, foreground=ACCENT_COLOR
        )

        # --- Header: title, light-mode note, badge links ---------------------
        header = ttk.Frame(root, style="Header.TFrame")
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)

        title_row = ttk.Frame(header, style="Header.TFrame")
        title_row.grid(row=0, column=0, sticky="w", padx=14, pady=(12, 0))
        title_font = tkfont.Font(family="Segoe UI", size=16, weight="bold")
        ttk.Label(
            title_row, text=PRODUCT_NAME, font=title_font, style="Title.Header.TLabel"
        ).pack(side="left")
        ttk.Label(
            title_row, text=f"   {SUBTITLE}", style="Subtitle.Header.TLabel"
        ).pack(side="left")

        ttk.Label(header, text=LIGHT_MODE_NOTE, style="Note.Header.TLabel").grid(
            row=1, column=0, sticky="w", padx=14, pady=(2, 8)
        )

        badges_row = ttk.Frame(header, style="Header.TFrame")
        badges_row.grid(row=2, column=0, sticky="w", padx=14, pady=(0, 12))
        for text, url, fill, hover_fill in BADGES:
            badge = _Badge(badges_row, text, url, fill, hover_fill, parent_bg=HEADER_BG)
            badge.pack(side="left", padx=(0, 8))

        ttk.Separator(header, orient="horizontal").grid(row=3, column=0, sticky="ew")

        # --- Inputs -----------------------------------------------------
        input_frame = ttk.LabelFrame(root, text="Input images")
        input_frame.grid(row=1, column=0, sticky="nsew", **pad)
        input_frame.columnconfigure(0, weight=1)
        input_frame.rowconfigure(0, weight=1)
        root.rowconfigure(1, weight=1)

        list_wrap = ttk.Frame(input_frame)
        list_wrap.grid(row=0, column=0, sticky="nsew", padx=(8, 0), pady=8)
        list_wrap.columnconfigure(0, weight=1)
        list_wrap.rowconfigure(0, weight=1)

        self.listbox = tk.Listbox(list_wrap, selectmode=tk.EXTENDED, height=8)
        self.listbox.grid(row=0, column=0, sticky="nsew")
        list_scroll_y = ttk.Scrollbar(list_wrap, orient="vertical", command=self.listbox.yview)
        list_scroll_y.grid(row=0, column=1, sticky="ns")
        list_scroll_x = ttk.Scrollbar(list_wrap, orient="horizontal", command=self.listbox.xview)
        list_scroll_x.grid(row=1, column=0, sticky="ew")
        self.listbox.configure(
            yscrollcommand=list_scroll_y.set, xscrollcommand=list_scroll_x.set
        )

        button_col = ttk.Frame(input_frame)
        button_col.grid(row=0, column=1, sticky="n", padx=8, pady=8)
        ttk.Button(button_col, text="Add files...", command=self._add_files).pack(
            fill="x", pady=2
        )
        ttk.Button(button_col, text="Add folder...", command=self._add_folder).pack(
            fill="x", pady=2
        )
        ttk.Button(button_col, text="Remove selected", command=self._remove_selected).pack(
            fill="x", pady=2
        )
        ttk.Button(button_col, text="Clear all", command=self._clear_all).pack(fill="x", pady=2)

        # --- Output -------------------------------------------------------
        out_frame = ttk.LabelFrame(root, text="Output directory")
        out_frame.grid(row=2, column=0, sticky="ew", **pad)
        out_frame.columnconfigure(0, weight=1)

        ttk.Entry(out_frame, textvariable=self.outdir_var).grid(
            row=0, column=0, sticky="ew", padx=(8, 4), pady=8
        )
        ttk.Button(out_frame, text="Browse...", command=self._choose_outdir).grid(
            row=0, column=1, padx=(0, 8), pady=8
        )
        ttk.Button(out_frame, text="Open folder", command=self._open_outdir).grid(
            row=0, column=2, padx=(0, 8), pady=8
        )

        # --- Model -----------------------------------------------------
        model_frame = ttk.LabelFrame(root, text="Model")
        model_frame.grid(row=3, column=0, sticky="ew", **pad)
        ttk.Radiobutton(
            model_frame, text="Base (default)", variable=self.model_var, value="base"
        ).pack(side="left", padx=8, pady=6)
        ttk.Radiobutton(
            model_frame, text="Body composition", variable=self.model_var, value="body_comp"
        ).pack(side="left", padx=8, pady=6)

        # --- Options -----------------------------------------------------
        options_frame = ttk.LabelFrame(root, text="Options")
        options_frame.grid(row=4, column=0, sticky="ew", **pad)

        ttk.Checkbutton(
            options_frame, text="Fast mode (default)", variable=self.fast_var
        ).pack(side="left", padx=8, pady=6)

        ttk.Label(options_frame, text="Split level:").pack(side="left", padx=(16, 4))
        for value, label in (("0", "0"), ("1", "1 (default)"), ("2", "2")):
            ttk.Radiobutton(
                options_frame, text=label, variable=self.split_level_var, value=value
            ).pack(side="left", padx=4, pady=6)

        # --- Run controls ---------------------------------------------------
        run_frame = ttk.Frame(root)
        run_frame.grid(row=5, column=0, sticky="ew", **pad)
        run_frame.columnconfigure(0, weight=1)

        self.run_button = ttk.Button(run_frame, text="Run", command=self._on_run)
        self.run_button.grid(row=0, column=1, padx=4)
        self.cancel_button = ttk.Button(
            run_frame, text="Cancel", command=self._on_cancel, state="disabled"
        )
        self.cancel_button.grid(row=0, column=2, padx=4)
        ttk.Label(run_frame, textvariable=self.overall_status_var).grid(
            row=0, column=0, sticky="w"
        )

        # --- Progress ---------------------------------------------------
        progress_frame = ttk.Frame(root)
        progress_frame.grid(row=6, column=0, sticky="ew", **pad)
        progress_frame.columnconfigure(0, weight=1)

        self.overall_progress = ttk.Progressbar(
            progress_frame, orient="horizontal", mode="determinate", maximum=100
        )
        self.overall_progress.grid(row=0, column=0, sticky="ew", pady=(0, 4))

        ttk.Label(progress_frame, textvariable=self.current_status_var).grid(
            row=1, column=0, sticky="w"
        )
        self.current_progress = ttk.Progressbar(
            progress_frame, orient="horizontal", mode="indeterminate", maximum=100
        )
        self.current_progress.grid(row=2, column=0, sticky="ew", pady=(2, 0))

        # --- Log ----------------------------------------------------------
        log_frame = ttk.LabelFrame(root, text="Log")
        log_frame.grid(row=7, column=0, sticky="nsew", **pad)
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        root.rowconfigure(7, weight=2)

        self.log_text = ScrolledText(log_frame, height=12, state="disabled", wrap="none")
        self.log_text.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)

    # ------------------------------------------------------------------
    # Input list management
    # ------------------------------------------------------------------
    def _add_files(self) -> None:
        patterns = " ".join(f"*{ext}" for ext in SUPPORTED_EXTENSIONS)
        paths = filedialog.askopenfilenames(
            title="Select input images",
            filetypes=[("Supported images", patterns), ("All files", "*.*")],
        )
        for path in paths:
            self._add_job(path)

    def _add_folder(self) -> None:
        path = filedialog.askdirectory(title="Select a folder (NIfTI images or a DICOM series)")
        if path:
            self._add_job(path)

    def _add_job(self, path: str) -> None:
        if any(job.path == path for job in self._jobs):
            return
        job = _Job(path)
        self._jobs.append(job)
        self.listbox.insert(tk.END, path)

    def _remove_selected(self) -> None:
        for index in reversed(self.listbox.curselection()):
            self.listbox.delete(index)
            del self._jobs[index]

    def _clear_all(self) -> None:
        self.listbox.delete(0, tk.END)
        self._jobs.clear()

    def _choose_outdir(self) -> None:
        path = filedialog.askdirectory(title="Select output directory")
        if path:
            self.outdir_var.set(path)

    def _open_outdir(self) -> None:
        outdir = self.outdir_var.get().strip()
        if not outdir or not os.path.isdir(outdir):
            messagebox.showinfo("MRSegmentator", "The output directory does not exist yet.")
            return
        if sys.platform == "win32":
            os.startfile(outdir)  # type: ignore[attr-defined]
        else:
            messagebox.showinfo("MRSegmentator", outdir)

    # ------------------------------------------------------------------
    # Run / cancel
    # ------------------------------------------------------------------
    def _on_run(self) -> None:
        if self._running:
            return
        if not self._jobs:
            messagebox.showerror("MRSegmentator", "Add at least one file or folder first.")
            return
        outdir = self.outdir_var.get().strip()
        if not outdir:
            messagebox.showerror("MRSegmentator", "Choose an output directory first.")
            return

        try:
            os.makedirs(outdir, exist_ok=True)
        except OSError as error:
            messagebox.showerror("MRSegmentator", f"Cannot create output directory:\n{error}")
            return

        self._running = True
        self._cancel_requested = False
        self.run_button.configure(state="disabled")
        self.cancel_button.configure(state="normal")
        self._clear_log()
        self.overall_progress.configure(value=0)
        self.overall_status_var.set(f"Running 0/{len(self._jobs)}")
        self.current_status_var.set("")

        jobs = list(self._jobs)
        model = self.model_var.get()
        split_level = self.split_level_var.get()
        fast = self.fast_var.get()
        self._worker = threading.Thread(
            target=self._run_batch, args=(jobs, outdir, model, split_level, fast), daemon=True
        )
        self._worker.start()

    def _on_cancel(self) -> None:
        if not self._running:
            return
        self._cancel_requested = True
        self.cancel_button.configure(state="disabled")
        self._append_log("--- cancel requested, stopping after the current file ---")
        if self._proc is not None and self._proc.poll() is None:
            try:
                self._proc.terminate()
            except OSError:
                pass

    def _on_close(self) -> None:
        if self._running:
            if not messagebox.askyesno(
                "MRSegmentator", "A run is in progress. Stop it and close the window?"
            ):
                return
            self._on_cancel()
        self.root.destroy()

    # ------------------------------------------------------------------
    # Worker thread: runs the unmodified CLI once per job, sequentially
    # ------------------------------------------------------------------
    def _run_batch(
        self, jobs: List[_Job], outdir: str, model: str, split_level: str, fast: bool
    ) -> None:
        total = len(jobs)
        succeeded = 0
        failed: List[str] = []
        index = 0

        for index, job in enumerate(jobs, start=1):
            if self._cancel_requested:
                break

            self._queue.put(("overall", (index - 1, total, job.name)))
            self._queue.put(("current_reset", job.name))

            args = ["--input", job.path, "--outdir", outdir, "--split_level", split_level]
            if fast:
                args.append("--fast")
            if model == "body_comp":
                args.append("--body_comp")

            command = _cli_command(args)
            self._queue.put(("line", f"$ mrsegmentator {' '.join(args)}"))

            try:
                proc = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    cwd=str(WINDOWS_DIR.parent),
                )
            except OSError as error:
                self._queue.put(("line", f"Failed to start: {error}"))
                failed.append(job.name)
                continue

            self._proc = proc
            assert proc.stdout is not None
            for raw_line in proc.stdout:
                line = raw_line.rstrip("\n")
                if not line:
                    continue
                self._queue.put(("line", line))
                match = PERCENT_RE.search(line)
                if match:
                    self._queue.put(("current_percent", int(match.group(1))))

            returncode = proc.wait()
            self._proc = None

            if self._cancel_requested and returncode != 0:
                failed.append(f"{job.name} (cancelled)")
                break
            elif returncode == 0:
                succeeded += 1
            else:
                failed.append(job.name)

        self._queue.put(("overall", (total if not self._cancel_requested else index, total, "")))
        self._queue.put(("done", (succeeded, failed)))

    # ------------------------------------------------------------------
    # GUI-thread queue polling
    # ------------------------------------------------------------------
    def _poll_queue(self) -> None:
        try:
            while True:
                kind, payload = self._queue.get_nowait()
                if kind == "line":
                    self._append_log(str(payload))
                elif kind == "overall":
                    done, total, name = payload  # type: ignore[misc]
                    pct = (done / total * 100) if total else 0
                    self.overall_progress.configure(value=pct)
                    if name:
                        self.overall_status_var.set(f"Running {done + 1}/{total}: {name}")
                    else:
                        self.overall_status_var.set(f"{done}/{total} finished")
                elif kind == "current_reset":
                    self.current_status_var.set(f"Processing: {payload}")
                    self.current_progress.configure(mode="indeterminate", value=0)
                    self.current_progress.start(80)
                elif kind == "current_percent":
                    self.current_progress.stop()
                    self.current_progress.configure(mode="determinate")
                    self.current_progress.configure(value=int(payload))  # type: ignore[arg-type]
                elif kind == "done":
                    self._on_batch_done(*payload)  # type: ignore[misc]
        except queue.Empty:
            pass
        self.root.after(100, self._poll_queue)

    def _on_batch_done(self, succeeded: int, failed: List[str]) -> None:
        self._running = False
        self.run_button.configure(state="normal")
        self.cancel_button.configure(state="disabled")
        self.current_progress.stop()
        self.current_progress.configure(mode="determinate", value=100 if not failed else 0)
        self.current_status_var.set("")

        if self._cancel_requested:
            self.overall_status_var.set("Cancelled")
            self._append_log("--- cancelled ---")
            return

        self.overall_status_var.set(f"Done: {succeeded} succeeded, {len(failed)} failed")
        self._append_log(f"--- finished: {succeeded} succeeded, {len(failed)} failed ---")
        if failed:
            messagebox.showwarning(
                "MRSegmentator",
                "Finished with errors:\n" + "\n".join(failed) + "\n\nSee the log for details.",
            )
        else:
            messagebox.showinfo("MRSegmentator", f"Finished. {succeeded} file(s) segmented.")

    # ------------------------------------------------------------------
    # Log pane
    # ------------------------------------------------------------------
    def _clear_log(self) -> None:
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", tk.END)
        self.log_text.configure(state="disabled")

    def _append_log(self, text: str) -> None:
        self.log_text.configure(state="normal")
        self.log_text.insert(tk.END, text + "\n")
        line_count = int(self.log_text.index("end-1c").split(".")[0])
        if line_count > MAX_LOG_LINES:
            self.log_text.delete("1.0", f"{line_count - MAX_LOG_LINES}.0")
        self.log_text.see(tk.END)
        self.log_text.configure(state="disabled")


def main() -> None:
    root = tk.Tk()
    _apply_window_icon(root)
    MRSegGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
