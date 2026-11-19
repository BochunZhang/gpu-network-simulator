# Days AGO maintained copy

Imported from [iQua/days at 048d48743119257fd8d4b2e5f0fb2228153fd955](https://github.com/iQua/days/tree/048d48743119257fd8d4b2e5f0fb2228153fd955) on 2026-11-19. The upstream source, Cargo workspace, configurations, tests, proofs and license are retained. Repository automation, editor settings and assistant configuration are omitted. The upstream version uses AGPL-3.0-only as documented in [LICENSE](LICENSE).

This copy includes the project's GB200 SM100 build and device-selection changes. The CUDA executor can be linked to the FT48 warp collector by setting `FT48_WARP_LIBDIR` at build time. It records graph replays after graph instantiation and compact-gather kernel launches during final readback when `FT48_WARP_OUTPUT` is set.

The FT48 Nsight Systems task sets `DAYS_FT48_CAPTURE_SECONDS` on the profiled CUDA run. Collection starts before the first graph replay and stops at a synchronized wave boundary. The simulator records the elapsed wall time, completed waves and committed rounds, then continues to the scenario endpoint.
