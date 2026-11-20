# FT48 warp lifetime traces

Open `warp/preview.perfetto.json` in [Perfetto](https://ui.perfetto.dev/). Each process track is one SM. Expand it to see the kernel, launch ordinal, CTA, warp and observed entry/exit times. The preview shows the first two CTAs of up to 45 launches of replay 0. The binary and its index retain all captured warp lifetimes. For example, to view every warp of CTA 55 in launch 8 of replay 12:

```sh
python3 profiling/warp_trace/trace.py export /path/to/warp replay-12.json --replay 12 --node 8 --cta 55
```

`--sm` filters the selected CTAs. Use `--max-ctas` and `--max-nodes` to expand the selection, or `--phase direct` to inspect an individual Days gather.

## Data layout

- `summary.json` records the simulation round count, replay, launch and warp counts, and coverage checks. Its `per_sm` array counts all observed warp instances and sums their elapsed intervals per SM. Multiple overlapping lifetimes contribute separately; this sum includes waits.
- `lifetimes.bin` contains uncompressed little-endian records (`<IHHQQ`, 24 bytes). Fields are linear CTA index, logical warp index within its CTA, SM index, start ns, and end ns. All FT48 grids are one-dimensional.
- `index.sqlite` maps a simulation replay and launch ordinal to its byte offset and kernel/launch geometry. Records within each launch are sorted by CTA and then warp. Each chunk is an independent IKET session and time domain. A launch retains its original SDK context ID, grid ID, array index and graph key (empty for the flat launch array). `sdk_node_id` retains the SDK graph node ID if supplied. The `node` column is a replay-local ordinal in observed start-time order. Simultaneous nodes can share a start time. Days graph events also expose the zero-based `encoded_attempt`; a padded attempt can enter the kernel and return early.
- `pending.iket.json` holds the raw window between session stop and successful compaction. On failure it remains available for diagnosis. A successful run removes it after the compact files have been written.

The index can be queried using SQLite:

```sql
SELECT l.node, k.short_name, l.grid_x, l.block_x, l.warp_count, l.start_ns, l.end_ns
FROM launches l JOIN kernels k ON k.id = l.kernel
WHERE l.phase = 'graph' AND l.replay = 0
ORDER BY l.node;
```

Lifetime intervals include stalls and waits. Overlap describes co-residency over the observed entry/exit intervals. Instruction issue order and CUDA graph dependency edges require other measurements. Compare timestamps within the same capture window. Separate sessions have independent time origins.

## Capture and checks

FT48 uses NVIDIA IKET's in-process lifetime-only sessions. The host builds the graph definition, starts IKET, and instantiates a new graph executable for that capture window. The final replay in the window synchronizes before IKET stops. Days uses one session per 64-attempt replay and separate sessions for its three graph-external gathers. GeDES batches sequential synchronized replays. The simulator-specific runners and `tools/warp_ft48.py` set up the capture.

The optional task parameter `warp_graph_replays` caps recording at the first specified number of graph replays. Both simulators continue through their normal completion gate, and Days still records the three direct gathers. The trace summary records the requested limit, the full simulation replay count derived from its completed rounds, and the number of recorded replays.

The decoder streams the location table and a kernel at a time, writes each launch's observed warp intervals, and commits a window only after checking every node's name, grid and block and every expected CTA/warp pair. A mismatched, overlapping, missing or duplicate interval identity fails the capture. Missing baseline or correctness gates are reported separately by the simulator-specific runner. Capture overhead, memory usage and total application time belong to the instrumented run.

[IKET JSON schema and timestamp scope](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/guides/iket_profiling.html#json-trace-shape)
