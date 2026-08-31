"""Evaluation code vendored verbatim from the challenge organisers.

Files in this package are **unmodified** copies from
https://github.com/lab-midas/autoPETV (commit 231d9a8, 2026-06-17), Apache License 2.0.
See the NOTICE file at the repository root.

They are copied rather than reimplemented on purpose. `metrics` is the official scorer, and a
reimplementation would quietly stop numbers being comparable with everyone else's. The scribble
rule in `simulate_scribbles` *is* the evaluation protocol, and it is subtle -- the choice between
a foreground and a background correction compares the length of the drawn scribbles, not the
volume of the errors -- so reproducing it from the paper would be guesswork.

Because they are unmodified, they are excluded from this repository's linting and formatting and
stay byte-for-byte diffable against upstream.
"""
