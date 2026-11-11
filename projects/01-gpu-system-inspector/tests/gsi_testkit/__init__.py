"""Internal test kit for GPU System Inspector: synthetic builders, FX fixtures and fakes.

Everything here is synthetic. Reports built from it have `data_origin = FIXTURE` and every
`Observed.method` starts with `fixture:`, so nothing can be mistaken for a measurement.
This package lives under `tests/` and is never shipped (DEC-17).
"""
