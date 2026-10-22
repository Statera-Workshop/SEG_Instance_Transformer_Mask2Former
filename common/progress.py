"""
Terminal progress bar (no extra dependency).

    ▶ [2/6] COMPLEX v3 | epoch 03/12 |██████████░░░░░░░░░░| 1234/2506  49% | loss 12.34 | 4.1 it/s | epoch 05:12 | total 1:12:30

The bar is redrawn in place on the terminal only. The log file receives one
plain summary line per `log_every` updates instead, so logs stay readable.
When the output is not a terminal, the bar falls back to those plain lines.
"""

import os
import sys
import time

from common import utils


def fmt_time(sec):
    sec = int(max(sec, 0))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def queue_tag():
    pos = os.environ.get("QUEUE_POS")
    return f"[{pos}] " if pos else ""


class Progress:
    def __init__(self, total, desc, width=24, log_every=100, extra_total=0):
        self.total = max(int(total), 1)
        self.desc = desc
        self.width = width
        self.log_every = log_every
        self.extra_total = extra_total   # items still to come after this bar (for the total ETA)
        self.start = time.time()
        self.last_draw = 0.0
        self.n = 0
        self.tty = sys.__stdout__.isatty() or os.environ.get("PROGRESS_FORCE") == "1"

    def _line(self, postfix):
        frac = self.n / self.total
        filled = int(self.width * frac)
        bar = "█" * filled + "░" * (self.width - filled)
        elapsed = time.time() - self.start
        rate = self.n / elapsed if elapsed > 0 else 0.0
        eta = (self.total - self.n) / rate if rate > 0 else 0.0
        parts = [f"{queue_tag()}{self.desc}", f"{bar} {self.n}/{self.total} {frac * 100:3.0f}%"]
        parts += [f"{k} {v}" for k, v in postfix.items()]
        parts.append(f"{rate:.1f} it/s")
        parts.append(f"left {fmt_time(eta)}")
        if self.extra_total:
            parts.append(f"total left {fmt_time(eta + self.extra_total / rate if rate > 0 else 0)}")
        return " | ".join(parts)

    def update(self, n=1, **postfix):
        self.n += n
        done = self.n >= self.total
        line = self._line(postfix)
        if self.tty:
            now = time.time()
            if done or now - self.last_draw > 0.5:
                sys.__stdout__.write("\r\033[K" + line)
                sys.__stdout__.flush()
                self.last_draw = now
            if self.n % self.log_every == 0 or done:
                utils.log_only(line)
        elif self.n % self.log_every == 0 or done:
            print(line)

    def close(self):
        if self.tty:
            sys.__stdout__.write("\n")
            sys.__stdout__.flush()
