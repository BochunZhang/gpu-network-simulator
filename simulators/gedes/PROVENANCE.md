# GeDES simulator sources

The initial source for this directory is the `GeDES_code/` tree and `README.md` from [mobinets/GeDES at 1376c638a0ae66116430f953da137d0c536675ad](https://github.com/mobinets/GeDES/tree/1376c638a0ae66116430f953da137d0c536675ad). The upstream commit includes the authors' build instructions and contains no separate LICENSE file. Their evaluation binaries, captured data and plotting scripts are outside this source snapshot.

The maintained source incorporates the project's FT48 build paths, fixed seed 41, flow-start queue sizing, completion readback, and terminal byte and queue audit. The optional `FT48_WARP_LIBDIR` CMake setting links the host replay hooks for the FT48 lifetime capture. In the profiled run `GEDES_FT48_GRAPH_DOT` exports the CUDA graph with its dependency edges before the first replay.
