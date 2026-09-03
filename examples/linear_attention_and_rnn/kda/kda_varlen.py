"""Host-side varlen bookkeeping shared by the reference layer and all six stages.

A leaf module on purpose: it imports nothing from this package, so both the CPU
goldens and the kernel host wrappers can depend on it without the kernels
acquiring a run-time dependency on the reference layer.

The frozen varlen contract (FlashAttention's, the one FLA follows)
------------------------------------------------------------------
    cu_seqlens    [N + 1] cumulative token counts, non-decreasing, first entry
                  0 and last entry the flattened token count
    sequence i    bos = cu_seqlens[i], eos = cu_seqlens[i + 1], T_i = eos - bos
    B             is 1 -- every sequence is flattened onto the token axis
    initial_state first dim is N, the sequence count, NOT a batch
    T_i == 0      legal.  An empty sequence is what a scheduler produces for an
                  idle slot; FLA's fused_recurrent returns early on one rather
                  than rejecting it, so the whole pipeline has to survive it.

Why the boundaries have to come back to the host
------------------------------------------------
A T.Kernel grid extent is a trace-time Python int -- it cannot be read out of
device memory.  The number of chunks in a varlen batch is
``sum_i ceil(T_i / C)``, which depends on the values inside cu_seqlens, so those
values have to be on the host before the grid can be sized.  That is one
device-to-host sync per call, and it is exactly why FLA carries a separate
``cu_seqlens_cpu`` argument.  Pass ``cu_seqlens`` already on the CPU to skip it.

What does NOT come back to the host is any token data.  The metadata built here
is O(N) plus O(NT_total) int32 -- for a 4096-token batch of 8 sequences at
C = 64 that is 64 chunks * 3 int32 = 768 bytes.  The acceptance gate forbids
host-side transforms that hide kernel cost; a kilobyte of loop bounds is not one.

Why the ragged chunk needs explicit metadata at all
---------------------------------------------------
For a fixed-length batch the framework gives tail handling away: every GM copy
is clamped by compute_valid_extent (src/op/ascend.cc) to ``shape - offset``, and
the only ragged chunk sits at the end of the tensor, so ``shape - offset`` is
already the right valid row count.

Under varlen that stops being true.  Sequence i's last chunk is ragged in the
MIDDLE of the flattened tensor, where ``shape - offset`` is the distance to the
end of the whole batch, not to eos.  Measured, not assumed -- an unbounded tile
copy at t0 = 64 with 6 valid rows was seen to write 58 rows past the end of its
sequence, straight over the next sequence's tokens (PROBES/probe_varlen2.log).
So every stage takes its valid row count from ``rows`` below and passes it as a
run-time extent on the copy, which the framework then honours and gap-fills with
zeros (PROBES/probe_varlen3.log, probe_varlen4.log).
"""

import torch

__all__ = ["varlen_bounds", "chunk_layout", "chunk_meta", "seq_meta", "META_COLS", "SEQ_META_COLS"]

# Column layout of the per-chunk metadata tensor.  One row per chunk of the
# whole batch, indexed by the flat block id, so a kernel block reads its own row
# and needs nothing else.
META_I_N = 0  # which sequence this chunk belongs to (indexes initial_state / final_state)
META_T0 = 1  # absolute first token of this chunk in the flattened batch
META_ROWS = 2  # valid rows, 1 .. C.  Less than C only on a sequence's last chunk.
META_COLS = 3

# Column layout of the per-sequence metadata tensor, for the one stage whose
# grid is per sequence rather than per chunk (stage 5 carries state serially
# across chunks, so its chunks cannot be independent blocks).
SEQ_BOS = 0
SEQ_EOS = 1
SEQ_CHUNK_OFF = 2  # first chunk slot of this sequence in the NT_total-long run
SEQ_META_COLS = 3


def varlen_bounds(cu_seqlens, q=None, k=None, v=None, g=None, beta=None, initial_state=None):
    """Validate the varlen contract and return [(bos, eos)] as plain ints.

    The tensors are optional so the reference layer, the kernel wrappers and the
    tests can all reuse the same validation with whatever they happen to hold.
    """
    assert cu_seqlens.dim() == 1, f"cu_seqlens is [N+1], got shape {tuple(cu_seqlens.shape)}"
    assert cu_seqlens.numel() >= 2, "cu_seqlens needs at least [0, T] -- one sequence"
    # Integral, explicitly.  The frozen contract says int64 caller-side, every
    # kernel takes int32, and FLA declares LongTensor and asserts nothing -- so
    # both integer widths are accepted here.  A FLOAT tensor is not: the
    # `int(x)` below would truncate it into perfectly plausible-looking bounds
    # and the only symptom would be wrong answers on some sequences.
    assert not cu_seqlens.is_floating_point(), f"cu_seqlens must be an integer tensor, got {cu_seqlens.dtype}"

    flat = [int(x) for x in cu_seqlens.tolist()]
    assert flat[0] == 0, f"cu_seqlens must start at 0, got {flat[0]}"

    bounds = []
    for i in range(len(flat) - 1):
        bos, eos = flat[i], flat[i + 1]
        assert eos >= bos, f"cu_seqlens must be non-decreasing; sequence {i} runs {bos}..{eos}"
        bounds.append((bos, eos))
    N = len(bounds)

    named = (("q", q), ("k", k), ("v", v), ("g", g), ("beta", beta))
    present = [(n, t) for n, t in named if t is not None]
    if present:
        T_total = present[0][1].shape[1]
        assert flat[-1] == T_total, f"cu_seqlens must end at the flattened token count {T_total}, got {flat[-1]}"
        for name, t in present:
            assert t.shape[0] == 1, f"varlen flattens every sequence onto the token axis, so {name} must have B == 1, got {t.shape[0]}"
            assert t.shape[1] == T_total, f"{name} has {t.shape[1]} tokens, cu_seqlens says {T_total}"
    if initial_state is not None:
        assert initial_state.shape[0] == N, f"initial_state's first dim is the sequence count {N}, got {initial_state.shape[0]}"

    return bounds


def chunk_layout(bounds, C):
    """(first chunk slot of each sequence, total chunk count).

    ``offsets[n] = sum_{m < n} ceil(T_m / C)`` and ``NT_total = sum_n ceil(T_n / C)``.
    Chunking restarts at every bos, so a sequence's chunk i covers tokens
    ``bos + i*C .. min(bos + (i+1)*C, eos)`` -- it is NOT aligned to a global
    multiple of C, which is the whole reason a chunk cannot find its own bounds
    from its block id alone.

    An empty sequence contributes zero slots and simply does not appear in the
    run; its offset still points at where it would have started, which keeps the
    following sequences correct.

    Worked example, cu_seqlens = [0, 70, 70, 200] at C = 64 -- the middle
    sequence is deliberately empty:

        sequence 0  T = 70   ceil(70/64) = 2 chunks   offset 0
        sequence 1  T = 0    0 chunks                 offset 2
        sequence 2  T = 130  ceil(130/64) = 3 chunks  offset 2
        NT_total = 5

        chunk slot   i_n   t0    rows
             0        0      0     64
             1        0     64      6   <- ragged, and INTERIOR to the batch
             2        2     70     64
             3        2    134     64
             4        2    198      2   <- ragged, at the end of the batch
    """
    offsets, total = [], 0
    for bos, eos in bounds:
        offsets.append(total)
        total += -(-(eos - bos) // C)
    return offsets, total


def chunk_meta(bounds, C, device):
    """Per-chunk metadata, [NT_total, 3] int32, indexed by the flat block id.

    Column meanings are the META_* constants above.  Built on the host because
    every value in it is needed to size the grid anyway.
    """
    rows = []
    for i_n, (bos, eos) in enumerate(bounds):
        T_i = eos - bos
        for i_t in range(-(-T_i // C)):
            t0 = bos + i_t * C
            rows.append((i_n, t0, min(C, eos - t0)))
    if not rows:
        # Every sequence empty.  An empty int32 tensor keeps the caller's shape
        # arithmetic uniform; the grid it sizes is zero blocks, which the host
        # wrappers short-circuit before they ever launch.
        return torch.zeros((0, META_COLS), dtype=torch.int32, device=device)
    return torch.tensor(rows, dtype=torch.int32, device=device)


def seq_meta(bounds, C, device):
    """Per-sequence metadata, [N, 3] int32: (bos, eos, first chunk slot).

    For the stage whose grid is per sequence.  It derives its own trip count as
    ``ceildiv(eos - bos, C)`` on the device, which is zero for an empty sequence
    -- so the chunk loop simply does not run and the final state falls out of the
    initial state untouched, with no special case anywhere.
    """
    offsets, _ = chunk_layout(bounds, C)
    rows = [(bos, eos, off) for (bos, eos), off in zip(bounds, offsets)]
    return torch.tensor(rows, dtype=torch.int32, device=device)


# ------------------------------------------------------------------- selftest
def _bounds_from(seqlens):
    cu, acc = [0], 0
    for n in seqlens:
        acc += n
        cu.append(acc)
    return varlen_bounds(torch.tensor(cu, dtype=torch.int32))


def test_worked_example():
    """The example spelled out in chunk_layout's docstring, checked value by value."""
    print("== worked example: cu_seqlens = [0, 70, 70, 200], C = 64 ==")
    bounds = _bounds_from([70, 0, 130])
    offsets, total = chunk_layout(bounds, 64)
    meta = chunk_meta(bounds, 64, "cpu")
    smeta = seq_meta(bounds, 64, "cpu")

    want_meta = [(0, 0, 64), (0, 64, 6), (2, 70, 64), (2, 134, 64), (2, 198, 2)]
    ok = offsets == [0, 2, 2] and total == 5
    ok &= [tuple(int(x) for x in r) for r in meta] == want_meta
    ok &= [tuple(int(x) for x in r) for r in smeta] == [(0, 70, 0), (70, 70, 2), (70, 200, 2)]

    print(f"  offsets={offsets} NT_total={total}   {'ok' if offsets == [0, 2, 2] and total == 5 else 'FAIL'}")
    for r in meta:
        i_n, t0, rows = (int(x) for x in r)
        print(f"    i_n={i_n} t0={t0:4d} rows={rows:3d}{'   <- ragged' if rows < 64 else ''}")
    print(f"  {'ok' if ok else 'FAIL'}")
    return ok


def test_invariants():
    """Properties that must hold for any batch, checked over a spread of shapes."""
    print("== invariants over a spread of batches ==")
    ok = True
    cases = [
        ([64, 64, 64], 64, "equal, chunk-aligned"),
        ([70, 33, 129], 64, "all ragged"),
        ([0, 0, 0], 64, "every sequence empty"),
        ([1], 64, "one token"),
        ([4096], 64, "one long sequence"),
        ([100, 28, 1, 0, 63], 32, "mixed, C = 32"),
    ]
    for seqlens, C, note in cases:
        bounds = _bounds_from(seqlens)
        offsets, total = chunk_layout(bounds, C)
        meta = chunk_meta(bounds, C, "cpu")

        good = meta.shape == (total, META_COLS)
        good &= total == sum(-(-n // C) for n in seqlens)
        # Every token of every sequence is covered exactly once, and no chunk
        # reaches past its own eos.
        covered = []
        for r in meta:
            i_n, t0, rows = (int(x) for x in r)
            bos, eos = bounds[i_n]
            good &= 1 <= rows <= C and bos <= t0 and t0 + rows <= eos
            covered.extend(range(t0, t0 + rows))
        good &= covered == sorted(covered)
        good &= len(covered) == len(set(covered))
        good &= len(covered) == sum(seqlens)
        # A chunk is short only when it is the last one of its sequence.
        for r in meta:
            i_n, t0, rows = (int(x) for x in r)
            if rows < C:
                good &= t0 + rows == bounds[i_n][1]
        ok &= good
        print(f"  {str(seqlens):22s} C={C:2d} NT_total={total:3d} offsets={str(offsets):18s} {'ok' if good else 'FAIL'}  {note}")
    return ok


def test_rejects_bad_input():
    """The contract is asserted, not assumed."""
    print("== malformed input is rejected ==")
    ok = True
    q = torch.zeros(1, 10, 1, 8)
    bad = [
        (torch.tensor([0.0, 5.0, 10.0]), "a float tensor (would truncate silently)"),
        (torch.tensor([1, 5, 10], dtype=torch.int32), "does not start at 0"),
        (torch.tensor([0, 7, 5], dtype=torch.int32), "not non-decreasing"),
        (torch.tensor([0], dtype=torch.int32), "fewer than two entries"),
        (torch.tensor([[0, 10]], dtype=torch.int32), "not 1-D"),
        (torch.tensor([0, 4, 9], dtype=torch.int32), "does not end at the token count"),
    ]
    for cu, note in bad:
        try:
            varlen_bounds(cu, q)
            raised = False
        except AssertionError:
            raised = True
        ok &= raised
        print(f"  {note:38s} {'rejected' if raised else 'ACCEPTED -- FAIL'}")

    # B != 1 is the mistake most likely to be made, so it gets its own check.
    try:
        varlen_bounds(torch.tensor([0, 5, 10], dtype=torch.int32), torch.zeros(2, 10, 1, 8))
        raised = False
    except AssertionError:
        raised = True
    ok &= raised
    print(f"  {'B != 1 under varlen':38s} {'rejected' if raised else 'ACCEPTED -- FAIL'}")
    return ok


def main():
    ok = True
    ok &= test_worked_example()
    print()
    ok &= test_invariants()
    print()
    ok &= test_rejects_bad_input()
    print()
    if not ok:
        raise SystemExit(1)
    print("Test Passed!")


if __name__ == "__main__":
    main()
