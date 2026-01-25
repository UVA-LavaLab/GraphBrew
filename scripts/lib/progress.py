#!/usr/bin/env python3
"""
Generic progress tracking and reporting utilities for GraphBrew.

This module provides:
- ProgressTracker: Visual progress reporting with phases, steps, and tables
- ConsoleColors: ANSI color codes for terminal output
- Timer utilities for tracking elapsed time
- Table formatting for structured output

Example usage:
    from lib.progress import ProgressTracker
    
    # Create tracker
    progress = ProgressTracker()
    
    # Show banner
    progress.banner("My Experiment", "Running tests")
    
    # Track phases
    progress.phase_start("Phase 1", "Processing graphs")
    for i, graph in enumerate(graphs):
        progress.step(i + 1, len(graphs), graph)
        # ... do work ...
    progress.phase_end("Processed all graphs")
    
    # Print final summary
    progress.final_summary()
"""

import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


class ConsoleColors:
    """ANSI color codes for terminal output."""
    
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    DIM = '\033[2m'
    END = '\033[0m'
    
    @classmethod
    def colorize(cls, text: str, color: str, enabled: bool = True) -> str:
        """Apply color to text if enabled."""
        if not enabled:
            return text
        color_code = getattr(cls, color.upper(), None)
        if color_code:
            return f"{color_code}{text}{cls.END}"
        return text
    
    @classmethod
    def strip_colors(cls, text: str) -> str:
        """Remove ANSI color codes from text."""
        import re
        return re.sub(r'\033\[[0-9;]*m', '', text)


class Timer:
    """Simple timer for tracking elapsed time."""
    
    def __init__(self):
        self.start_time = time.time()
        self.checkpoints: Dict[str, float] = {}
    
    def elapsed(self) -> float:
        """Get elapsed time in seconds."""
        return time.time() - self.start_time
    
    def elapsed_formatted(self) -> str:
        """Get elapsed time as formatted string."""
        return format_duration(self.elapsed())
    
    def checkpoint(self, name: str):
        """Record a checkpoint."""
        self.checkpoints[name] = time.time()
    
    def since_checkpoint(self, name: str) -> float:
        """Get time since a checkpoint."""
        if name in self.checkpoints:
            return time.time() - self.checkpoints[name]
        return 0.0
    
    def reset(self):
        """Reset the timer."""
        self.start_time = time.time()
        self.checkpoints.clear()


def format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable string."""
    if seconds < 0:
        return "0s"
    
    mins, secs = divmod(int(seconds), 60)
    hours, mins = divmod(mins, 60)
    days, hours = divmod(hours, 24)
    
    if days > 0:
        return f"{days}d {hours}h {mins}m"
    elif hours > 0:
        return f"{hours}h {mins}m {secs}s"
    elif mins > 0:
        return f"{mins}m {secs}s"
    return f"{secs}s"


class ProgressTracker:
    """
    Visual progress reporting for multi-phase experiments.
    
    Provides:
    - Banners for major sections
    - Phase tracking with timing
    - Step progress bars
    - Substep status indicators
    - Table formatting
    - Statistics collection
    - Final summary reporting
    """
    
    def __init__(self, use_colors: bool = True, output_file: Optional[str] = None):
        """
        Initialize progress tracker.
        
        Args:
            use_colors: Enable ANSI colors (auto-disabled if not a TTY)
            output_file: Optional file to write progress to
        """
        self.use_colors = use_colors and sys.stdout.isatty()
        self.output_file = output_file
        self.timer = Timer()
        
        # Phase tracking
        self.current_phase: Optional[str] = None
        self.phase_start_time: Optional[float] = None
        self.phase_history: List[Dict[str, Any]] = []
        
        # Statistics
        self.stats: Dict[str, Any] = {
            'items_processed': 0,
            'total_time': 0,
            'best_results': {},
            'errors': [],
            'warnings': [],
        }
        
        # Output file handle
        self._file_handle = None
        if output_file:
            self._file_handle = open(output_file, 'w')
    
    def __del__(self):
        """Clean up file handle if open."""
        if self._file_handle:
            self._file_handle.close()
    
    def _output(self, text: str, end: str = '\n', flush: bool = False):
        """Write output to console and optionally file."""
        print(text, end=end, flush=flush)
        if self._file_handle:
            # Strip colors for file output
            clean_text = ConsoleColors.strip_colors(text)
            self._file_handle.write(clean_text + end)
            if flush:
                self._file_handle.flush()
    
    def _color(self, text: str, color: str) -> str:
        """Apply color to text if colors enabled."""
        return ConsoleColors.colorize(text, color, self.use_colors)
    
    def _elapsed(self) -> str:
        """Get elapsed time since start."""
        return self.timer.elapsed_formatted()
    
    def _phase_elapsed(self) -> str:
        """Get elapsed time since phase start."""
        if self.phase_start_time is None:
            return "0s"
        return format_duration(time.time() - self.phase_start_time)
    
    # ─────────────────────────────────────────────────────────────────────────
    # Banners and Headers
    # ─────────────────────────────────────────────────────────────────────────
    
    def banner(self, title: str, subtitle: str = None, width: int = 70):
        """
        Print a large banner for major sections.
        
        Args:
            title: Main banner title
            subtitle: Optional subtitle
            width: Banner width in characters
        """
        self._output("")
        self._output("╔" + "═" * (width - 2) + "╗")
        title_padded = title.center(width - 4)
        self._output("║ " + self._color(title_padded, 'BOLD') + " ║")
        if subtitle:
            sub_padded = subtitle.center(width - 4)
            self._output("║ " + self._color(sub_padded, 'CYAN') + " ║")
        self._output("╚" + "═" * (width - 2) + "╝")
        self._output("")
    
    def section(self, title: str, width: int = 70):
        """Print a section header."""
        self._output("")
        self._output("┌" + "─" * (width - 2) + "┐")
        self._output("│ " + self._color(title.ljust(width - 4), 'BOLD') + " │")
        self._output("└" + "─" * (width - 2) + "┘")
    
    def separator(self, char: str = "─", width: int = 70):
        """Print a separator line."""
        self._output(char * width)
    
    # ─────────────────────────────────────────────────────────────────────────
    # Phase Tracking
    # ─────────────────────────────────────────────────────────────────────────
    
    def phase_start(self, phase_name: str, description: str = None):
        """
        Start a new phase with visual indicator.
        
        Args:
            phase_name: Name of the phase
            description: Optional description
        """
        if self.current_phase:
            # End previous phase
            self.phase_history.append({
                'name': self.current_phase,
                'duration': time.time() - self.phase_start_time
            })
        
        self.current_phase = phase_name
        self.phase_start_time = time.time()
        
        self._output("")
        self._output("┌" + "─" * 68 + "┐")
        header = f"  PHASE: {phase_name}"
        self._output("│" + self._color(header.ljust(68), 'HEADER') + "│")
        if description:
            self._output("│  " + description.ljust(66) + "│")
        self._output("│  " + f"Started at: {datetime.now().strftime('%H:%M:%S')}".ljust(66) + "│")
        self._output("└" + "─" * 68 + "┘")
    
    def phase_end(self, summary: str = None):
        """
        End current phase with summary.
        
        Args:
            summary: Optional summary text
        """
        duration = self._phase_elapsed()
        
        # Record in history
        if self.current_phase:
            self.phase_history.append({
                'name': self.current_phase,
                'duration': time.time() - self.phase_start_time
            })
        
        self._output("")
        self._output("─" * 70)
        status = f"✓ Phase '{self.current_phase}' completed in {duration}"
        self._output(self._color(status, 'GREEN'))
        if summary:
            self._output(f"  {summary}")
        self._output("─" * 70)
        
        self.current_phase = None
        self.phase_start_time = None
    
    def phase_skip(self, reason: str = None):
        """Skip current phase with optional reason."""
        self._output("")
        status = f"⊘ Phase '{self.current_phase}' skipped"
        if reason:
            status += f": {reason}"
        self._output(self._color(status, 'YELLOW'))
        self.current_phase = None
        self.phase_start_time = None
    
    # ─────────────────────────────────────────────────────────────────────────
    # Progress Steps
    # ─────────────────────────────────────────────────────────────────────────
    
    def step(self, current: int, total: int, item_name: str, 
             extra_info: str = None, bar_width: int = 30):
        """
        Show progress for a numbered step.
        
        Args:
            current: Current step number (1-indexed)
            total: Total number of steps
            item_name: Name of current item
            extra_info: Optional extra information
            bar_width: Width of progress bar
        """
        pct = (current / total * 100) if total > 0 else 0
        filled = int(bar_width * current / total) if total > 0 else 0
        bar = "█" * filled + "░" * (bar_width - filled)
        
        line = f"  [{current:3d}/{total:3d}] [{bar}] {pct:5.1f}%  {item_name}"
        if extra_info:
            line += f"  │ {extra_info}"
        
        # Use \r for same-line updates when not at 100%
        end = '\n' if current == total else '\r'
        self._output(line.ljust(100), end=end, flush=True)
    
    def substep(self, message: str, status: str = "..."):
        """
        Show a substep with status indicator.
        
        Args:
            message: Substep message
            status: Status string (OK, DONE, SKIP, FAIL, WARN, ...)
        """
        status_colors = {
            '...': 'YELLOW',
            'OK': 'GREEN',
            'DONE': 'GREEN',
            'SKIP': 'CYAN',
            'FAIL': 'RED',
            'WARN': 'YELLOW',
            'RUN': 'BLUE',
        }
        
        status_color = status_colors.get(status, 'END')
        status_str = self._color(f"[{status:4s}]", status_color)
        self._output(f"    {status_str} {message}")
    
    def item(self, name: str, value: Any = None, indent: int = 0):
        """Print an item with optional value."""
        prefix = "  " * indent + "• "
        if value is not None:
            self._output(f"{prefix}{name}: {value}")
        else:
            self._output(f"{prefix}{name}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # Messages
    # ─────────────────────────────────────────────────────────────────────────
    
    def info(self, message: str, indent: int = 0):
        """Print an info message."""
        prefix = "  " * indent + "→ "
        self._output(f"{prefix}{message}")
    
    def success(self, message: str):
        """Print a success message."""
        self._output(self._color(f"  ✓ {message}", 'GREEN'))
    
    def warning(self, message: str, record: bool = True):
        """Print a warning message."""
        self._output(self._color(f"  ⚠ {message}", 'YELLOW'))
        if record:
            self.stats['warnings'].append(message)
    
    def error(self, message: str, record: bool = True):
        """Print an error message."""
        self._output(self._color(f"  ✗ {message}", 'RED'))
        if record:
            self.stats['errors'].append(message)
    
    def debug(self, message: str):
        """Print a debug message (dimmed)."""
        self._output(self._color(f"  [DEBUG] {message}", 'DIM'))
    
    # ─────────────────────────────────────────────────────────────────────────
    # Tables
    # ─────────────────────────────────────────────────────────────────────────
    
    def table_header(self, columns: List[Tuple[str, int]]):
        """
        Print a table header with column widths.
        
        Args:
            columns: List of (column_name, width) tuples
        """
        header = "  │ "
        sep = "  ├─"
        for name, width in columns:
            header += name.center(width) + " │ "
            sep += "─" * (width + 2) + "┼─"
        sep = sep[:-2] + "┤"
        self._output(sep)
        self._output(header)
        self._output(sep)
    
    def table_row(self, values: List[Tuple[str, int]], highlight: bool = False):
        """
        Print a table row.
        
        Args:
            values: List of (value, width) tuples
            highlight: Whether to highlight the row
        """
        row = "  │ "
        for val, width in values:
            cell = str(val).center(width)
            if highlight:
                cell = self._color(cell, 'GREEN')
            row += cell + " │ "
        self._output(row)
    
    def table_footer(self, columns: List[Tuple[str, int]]):
        """Print table footer line."""
        footer = "  └─"
        for _, width in columns:
            footer += "─" * (width + 2) + "┴─"
        footer = footer[:-2] + "┘"
        self._output(footer)
    
    # ─────────────────────────────────────────────────────────────────────────
    # Statistics
    # ─────────────────────────────────────────────────────────────────────────
    
    def stats_box(self, title: str, stats: Dict[str, Any], width: int = 50):
        """
        Print a summary box with statistics.
        
        Args:
            title: Box title
            stats: Dictionary of stats to display
            width: Box width
        """
        self._output("")
        self._output("  ┌" + "─" * width + "┐")
        self._output("  │ " + self._color(title.center(width - 2), 'BOLD') + " │")
        self._output("  ├" + "─" * width + "┤")
        for key, value in stats.items():
            line = f"  {key}: {value}"
            self._output("  │ " + line.ljust(width - 2) + " │")
        self._output("  └" + "─" * width + "┘")
    
    def record_stat(self, key: str, value: Any):
        """Record a custom statistic."""
        self.stats[key] = value
    
    def increment_stat(self, key: str, amount: int = 1):
        """Increment a counter statistic."""
        if key not in self.stats:
            self.stats[key] = 0
        self.stats[key] += amount
    
    def record_result(self, category: str, key: str, value: Any, 
                      compare: str = 'max'):
        """
        Record a result, keeping best value.
        
        Args:
            category: Result category
            key: Result key
            value: Result value
            compare: Comparison mode ('max' or 'min')
        """
        if 'best_results' not in self.stats:
            self.stats['best_results'] = {}
        
        cat = self.stats['best_results'].setdefault(category, {})
        if key not in cat:
            cat[key] = value
        else:
            if compare == 'max' and value > cat[key]:
                cat[key] = value
            elif compare == 'min' and value < cat[key]:
                cat[key] = value
    
    # ─────────────────────────────────────────────────────────────────────────
    # Final Summary
    # ─────────────────────────────────────────────────────────────────────────
    
    def final_summary(self, custom_stats: Dict[str, Any] = None):
        """
        Print final summary of the entire experiment.
        
        Args:
            custom_stats: Additional stats to include in summary
        """
        total_time = self.timer.elapsed()
        
        self.banner("EXPERIMENT COMPLETE", f"Total time: {format_duration(total_time)}")
        
        # Phase summary
        if self.phase_history:
            self._output("Phase Summary:")
            for phase in self.phase_history:
                dur = format_duration(phase['duration'])
                self._output(f"  • {phase['name']}: {dur}")
        
        # Best results
        if self.stats.get('best_results'):
            self._output("\nBest Results:")
            for category, results in self.stats['best_results'].items():
                self._output(f"  {category}:")
                for key, value in sorted(results.items()):
                    if isinstance(value, float):
                        self._output(f"    • {key}: {value:.4f}")
                    else:
                        self._output(f"    • {key}: {value}")
        
        # Custom stats
        if custom_stats:
            self._output("\nStatistics:")
            for key, value in custom_stats.items():
                self._output(f"  • {key}: {value}")
        
        # Warnings
        if self.stats.get('warnings'):
            count = len(self.stats['warnings'])
            self._output(self._color(f"\nWarnings: {count}", 'YELLOW'))
            for warn in self.stats['warnings'][:5]:
                self._output(f"  - {warn}")
            if count > 5:
                self._output(f"  ... and {count - 5} more")
        
        # Errors
        if self.stats.get('errors'):
            count = len(self.stats['errors'])
            self._output(self._color(f"\nErrors: {count}", 'RED'))
            for err in self.stats['errors'][:5]:
                self._output(f"  - {err}")
            if count > 5:
                self._output(f"  ... and {count - 5} more")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get all collected statistics."""
        return {
            **self.stats,
            'total_time': self.timer.elapsed(),
            'phases': self.phase_history.copy(),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Convenience functions
# ─────────────────────────────────────────────────────────────────────────────

def create_progress(use_colors: bool = True, 
                    output_file: str = None) -> ProgressTracker:
    """Create a new progress tracker instance."""
    return ProgressTracker(use_colors=use_colors, output_file=output_file)


# Global progress tracker for simple scripts
_global_progress: Optional[ProgressTracker] = None


def get_progress() -> ProgressTracker:
    """Get or create the global progress tracker."""
    global _global_progress
    if _global_progress is None:
        _global_progress = ProgressTracker()
    return _global_progress


def reset_progress():
    """Reset the global progress tracker."""
    global _global_progress
    _global_progress = None


__all__ = [
    'ConsoleColors',
    'Timer',
    'ProgressTracker',
    'format_duration',
    'create_progress',
    'get_progress',
    'reset_progress',
]
