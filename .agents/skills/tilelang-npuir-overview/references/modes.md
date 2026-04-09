# Developer and Expert Modes

## Developer mode

Use for faster implementation and easier maintenance.
Typical style:
- alloc_shared and alloc_fragment
- T.Parallel and reduce helpers
- concise kernel logic

## Expert mode

Use for fine control and performance tuning.
Typical style:
- explicit Scope blocks such as Scope("Cube") and Scope("Vector")
- explicit memory hierarchy and synchronization
- load_nd2nz and store_fixpipe in cube pipelines

## Rule of thumb

- Start with Developer mode for correctness
- Move hotspots to Expert mode after profiling
