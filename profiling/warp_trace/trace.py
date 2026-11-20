"""Stream IKET lifetime-only windows into an indexed, little-endian FT48 trace."""
import argparse
from array import array
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import sqlite3
import struct

RECORD = struct.Struct("<IHHQQ")  # CTA, warp within CTA, SM, trace-local start/end ns
DAYS = [
    ("days_horizon_sweep", 128, 1024), ("days_horizon", 1, 1024),
    ("days_round_reset", 576, 256), ("days_round_prepare", 576, 256),
    ("days_round", 576, 256), ("days_round_control_sweep", 128, 1024),
    ("days_round_control", 1, 1024), ("days_exchange_prefix_sweep", 128, 1024),
    ("days_exchange_prefix", 1, 1024), ("days_exchange_scatter", 576, 256),
    ("days_exchange_merge", 576, 256), ("days_round_finalize", 1, 1024),
]
GEDES = {
    ("DeviceScanInitKernel", 1, 128): 6, ("DeviceScanKernel", 4, 384): 6,
    ("recycle_packets", 864, 32): 3, ("update_pool_size_recycle", 1, 1): 3,
    ("allocate_packets", 864, 32): 3, ("update_pool_size_alloc", 1, 1): 3,
    ("DeviceReduceSingleTileKernel", 1, 512): 3, ("forward_frames", 576, 96): 2,
    ("routing_ipv4_packets", 864, 32): 2, ("forward_ipv4_packets", 864, 32): 2,
    ("DeviceReduceKernel", 8, 512): 2, ("decapsulate_frame", 864, 32): 1,
    ("forward_frames", 288, 96): 1, ("ipv4_decapsulation", 864, 32): 1,
    ("receive_tcp_packets", 864, 32): 1, ("send_tcp_packets", 864, 32): 1,
    ("ipv4_encapsulation", 864, 32): 1, ("encapsulate_frames", 864, 32): 1,
    ("p2p_transmit_frames", 5184, 32): 1, ("DeviceReduceKernel", 2, 512): 1,
    ("update_timer", 1, 1): 1,
}
GATHER = "days_compact_gather"
GRAPH_KEY = re.compile(r"graph_exec_\d+:\d+\Z")
UNKNOWN_GRAPH_SUFFIX = ":4294967295"  # IKET prints its failed CUDA ID lookup (-1) as uint32.
SCHEMA = """
create table if not exists chunks(
 id integer primary key, phase text, first_replay integer, replay_count integer,
 byte_start integer, byte_end integer, simulator text, format_version integer);
create table if not exists sm_totals(
 chunk integer, sm integer, warps integer, elapsed_ns integer, primary key(chunk,sm)) without rowid;
create table if not exists kernels(
 id integer primary key, name text unique not null, short_name text);
create table if not exists launches(
 phase text, replay integer, node integer, chunk integer, graph_key text,
 source_index integer, context_id text, grid_id text, kernel integer,
 grid_x integer, block_x integer, warp_count integer, byte_offset integer,
 start_ns integer, end_ns integer, sdk_node_id text, primary key(phase,replay,node)) without rowid;
"""


def require(ok, message):
    if not ok:
        raise ValueError(message)


def stream_locations(path):
    # The location table can precede or follow launches. Keep two uint32
    # columns per location, even for a replay with millions of warp records.
    import ijson
    ctas, ids = array("I"), array("I")
    with path.open("rb") as raw:
        events = iter(ijson.parse(raw))
        for prefix, event, value in events:
            if prefix == "locationTable" and event == "start_array":
                break
        else:
            raise ValueError("missing locationTable")
        for prefix, event, value in events:
            if prefix == "locationTable" and event == "end_array":
                break
            if prefix == "locationTable.item" and event == "start_map":
                builder = ijson.ObjectBuilder()
                builder.event(event, value)
                for prefix, event, value in events:
                    builder.event(event, value)
                    if prefix == "locationTable.item" and event == "end_map":
                        break
                loc = builder.value
                cta = loc["ctaId"]
                sm, warp = loc["smId"], loc["warpId"]
                require(len(cta) == 3 and cta[1:] == [0, 0], "unexpected non-1D FT48 CTA")
                require(0 <= sm < 65536 and 0 <= warp < 32, "invalid SM or logical warp")
                ctas.append(cta[0])
                ids.append((sm << 16) | warp)
    return ctas, ids


def stream_launches(path):
    import ijson
    with path.open("rb") as raw:
        events = iter(ijson.parse(raw))
        for prefix, event, value in events:
            if event != "start_map":
                continue
            if prefix == "launches.item":
                graph_key = ""
            elif prefix.startswith("graphLaunches.") and prefix.count(".") == 2 and prefix.endswith(".item"):
                graph_key = prefix.split(".")[1]
                require(GRAPH_KEY.fullmatch(graph_key), f"unknown graph key {graph_key}")
            else:
                continue
            builder = ijson.ObjectBuilder()
            builder.event(event, value)
            for field, event, value in events:
                builder.event(event, value)
                if field == prefix and event == "end_map":
                    break
            yield graph_key, builder.value


def store_launch(out, locations, launch, sm_totals=None):
    require(all(launch[f"{kind}Dim{axis}"] == 1 for kind in ("grid", "block") for axis in "YZ"),
            "unexpected non-1D FT48 launch")
    grid, block = launch["gridDimX"], launch["blockDimX"]
    warps = (block + 31) // 32
    size = grid * warps
    intervals = launch["warpLifetimes"]
    require(grid > 0 and 0 < block <= 1024 and len(intervals) == size,
            f"{launch['kernelName']}: expected {size} warp lifetimes, got {len(intervals)}")
    records = bytearray(size * RECORD.size)
    seen = bytearray(size)
    ctas, ids = locations
    first, last = None, 0
    for record in intervals:
        loc = record["locIdx"]
        if not 0 <= loc < len(ctas):
            raise ValueError("warp references unknown location")
        cta, packed = ctas[loc], ids[loc]
        warp, sm = packed & 0xffff, packed >> 16
        slot = cta * warps + warp
        if cta >= grid or warp >= warps or seen[slot]:
            raise ValueError(f"out-of-bounds or duplicate CTA/warp {cta}/{warp}")
        seen[slot] = 1
        start, end = record["startTs"], record["endTs"]
        if not 0 <= start <= end:
            raise ValueError("invalid warp lifetime")
        if sm_totals is not None:
            totals = sm_totals[sm]
            totals[0] += 1
            totals[1] += end - start
        first = start if first is None or start < first else first
        last = max(last, end)
        RECORD.pack_into(records, slot * RECORD.size, cta, warp, sm, start, end)
    offset = out.tell()
    out.write(records)
    return grid, block, size, offset, first, last


def canonical_name(name, simulator, phase):
    options = {GATHER} if phase == "direct" else {
        row[0] for row in (DAYS if simulator == "days" else GEDES)
    }
    # CUDA C++ entry points may be Itanium-mangled. A mangled name encodes
    # every identifier's length; matching the length avoids suffix collisions.
    for short in sorted(options, key=len, reverse=True):
        if name == short or (name.startswith("_Z") and f"{len(short)}{short}" in name):
            return short
        if re.search(r"(?<![A-Za-z0-9_])" + re.escape(short) + r"(?![A-Za-z0-9_])", name):
            return short
    raise ValueError(f"unexpected FT48 kernel {name!r}")


def validate_geometry(rows, simulator, phase):
    observed = [(r[0], r[5], r[6]) for r in rows]
    if phase == "direct":
        require(simulator == "days" and len(rows) == 1 and observed[0] in
                ((GATHER, 576, 256), (GATHER, 93, 256)), "unexpected direct launch")
    elif simulator == "days":
        require(observed == DAYS * 64, "Days graph does not match its ordered 64 x 12 attempts")
    else:
        require(Counter(observed) == GEDES, "GeDES graph does not match its 45 expected nodes")


def ingest(directory, simulator, phase, first, count):
    directory = Path(directory)
    raw = directory / "pending.iket.json"
    require(simulator in ("days", "gedes") and phase in ("graph", "direct") and count >= 1 and first >= 0,
            "bad capture window")
    locations = stream_locations(raw)
    groups = defaultdict(list)
    graph_metadata = defaultdict(list)
    empty_graph_keys = set()
    with sqlite3.connect(directory / "index.sqlite") as db, (directory / "lifetimes.bin").open("ab") as out:
        db.executescript(SCHEMA)
        offset = out.tell()
        expected = db.execute("select coalesce(max(byte_end),0) from chunks").fetchone()[0]
        require(offset == expected, "binary tail disagrees with committed capture windows")
        try:
            chunk = db.execute("insert into chunks(phase,first_replay,replay_count,byte_start,simulator,format_version) "
                               "values(?,?,?,?,?,1)", (phase, first, count, offset, simulator)).lastrowid
            kernels = {}
            sm_totals = defaultdict(lambda: [0, 0])
            for key, launch in stream_launches(raw):
                require(phase == "graph" or not key, "unexpected graph launch in direct capture")
                if key:
                    graph_metadata[key].append({field: value for field, value in launch.items()
                                                if field != "warpLifetimes"})
                    if key.endswith(UNKNOWN_GRAPH_SUFFIX) and launch["warpLifetimes"] == []:
                        empty_graph_keys.add(key)
                        continue
                name = launch["kernelName"]
                if name not in kernels:
                    short = canonical_name(name, simulator, phase)
                    db.execute("insert or ignore into kernels(name,short_name) values(?,?)", (name, short))
                    ident, = db.execute("select id from kernels where name=?", (name,)).fetchone()
                    kernels[name] = short, ident
                short, ident = kernels[name]
                geometry = store_launch(out, locations, launch, sm_totals)
                source_index = len(groups[key])
                sdk_node = next((str(launch[k]) for k in ("graphNodeId", "nodeId") if k in launch), None)
                groups[key].append((short, source_index, str(launch["contextId"]), str(launch["gridId"]),
                                    ident, *geometry, sdk_node, key))
            # A failed IKET graph ID lookup can leave an empty copy of an entire
            # graph group. Require a one-to-one match to validated launch metadata.
            for key in sorted(empty_graph_keys):
                prefix = key.removesuffix(UNKNOWN_GRAPH_SUFFIX) + ":"
                matches = sum(other.startswith(prefix) and graph_metadata[other] == graph_metadata[key]
                              for other in groups if other not in empty_graph_keys)
                require(key not in groups and matches == 1,
                        f"empty IKET graph group {key} has no unique populated copy")
            # The SDK may use one graph key per replay, combine multiple replays
            # under a graph-exec key, split nested graph keys, or place graph
            # nodes in the flat array. Keys are stored per launch.
            # The host synchronizes each replay, so their time envelopes must
            # be disjoint. Split only after verifying the exact node geometry.
            width = 1 if phase == "direct" else (768 if simulator == "days" else 45)
            ordered = sorted((r for group in groups.values() for r in group), key=lambda r: (r[9],r[1]))
            require(len(ordered) == width * count,
                    f"expected {width*count} graph/direct nodes, found {len(ordered)}")
            previous_end = None
            for batch_index in range(count):
                rows = ordered[batch_index*width:(batch_index+1)*width]
                require(previous_end is None or previous_end <= rows[0][9],
                        "ambiguous replay assignment: synchronized replay time envelopes overlap")
                previous_end = max(r[10] for r in rows)
                validate_geometry(rows, simulator, phase)
                db.executemany("insert into launches values(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                               ((phase, first + batch_index, node, chunk, row[-1], *row[1:-1])
                                for node, row in enumerate(rows)))
            db.executemany("insert into sm_totals values(?,?,?,?)",
                           ((chunk, sm, *stats) for sm, stats in sm_totals.items()))
            db.execute("update chunks set byte_end=? where id=?", (out.tell(), chunk))
            out.flush()
            db.commit()
        except BaseException:
            out.truncate(offset)
            raise


def verify(directory, simulator, rounds, graph_replays=None):
    directory = Path(directory)
    require(simulator in ("days", "gedes") and rounds > 0, "invalid tracking workload")
    require(graph_replays is None or (type(graph_replays) is int and graph_replays > 0),
            "graph replay limit must be a positive integer")
    simulation_replays = (rounds + 63) // 64 if simulator == "days" else rounds
    replays = simulation_replays if graph_replays is None else min(simulation_replays, graph_replays)
    require((directory / "index.sqlite").is_file(), "missing warp trace index")
    with sqlite3.connect(directory / "index.sqlite") as db:
        sizes = {"graph": 0, "direct": 0}
        chunks = expected_bytes = 0
        for phase, start, count, begin, end, name, version in db.execute("select phase,first_replay,replay_count,"
                "byte_start,byte_end,simulator,format_version from chunks order by id"):
            require(version == 1 and name == simulator and phase in sizes and start == sizes[phase]
                    and count > 0 and begin == expected_bytes and end > begin, "invalid or missing warp chunk")
            sizes[phase] += count
            chunks += 1
            expected_bytes = end
        require(sizes == {"graph": replays, "direct": 3 if simulator == "days" else 0},
                f"expected {replays} graph replays, got {sizes}")
        if simulator == "days":
            grids = [r[0] for r in db.execute("select grid_x from launches where phase='direct'")]
            require(sorted(grids) == [93, 576, 576], "missing Days compact gathers")
        count, warps = db.execute("select count(*),sum(warp_count) from launches").fetchone()
        nodes, per_replay = (768, 2269184) if simulator == "days" else (45, 23983)
        require(count == replays * nodes + sizes["direct"] and
                warps == replays * per_replay + (9960 if simulator == "days" else 0),
                "FT48 warp or launch coverage failed")
        require(expected_bytes == warps * RECORD.size == (directory / "lifetimes.bin").stat().st_size,
                "unexpected warp data length")
        per_sm = [dict(sm=sm, warp_lifetimes=n, summed_warp_ns=ns)
                  for sm, n, ns in db.execute("select sm,sum(warps),sum(elapsed_ns) "
                                              "from sm_totals group by sm order by sm")]
        require(sum(row["warp_lifetimes"] for row in per_sm) == warps, "SM counts disagree with warp records")
    summary = dict(per_sm=per_sm, backend="iket-lifetime-only", format_version=1, record_bytes=RECORD.size,
                   simulator=simulator, simulation_rounds=rounds, replays=sizes["graph"],
                   direct_launches=sizes["direct"],
                   capture_windows=chunks, kernel_launches=count, warp_lifetimes=warps,
                   binary_bytes=expected_bytes, coverage_passed=True, clock_domain="per IKET capture window")
    if graph_replays is not None:
        summary.update(requested_graph_replays=graph_replays, simulation_graph_replays=simulation_replays)
    (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    export(directory, "graph", 0, directory / "preview.perfetto.json", max_nodes=45, max_ctas=2)
    return summary


def export(directory, phase, replay, destination, node=None, sm=None, max_nodes=45, max_ctas=2, cta=None):
    directory = Path(directory)
    require(max_nodes > 0 and max_ctas > 0 and (cta is None or cta >= 0), "invalid export bounds")
    events = []
    processes = set()
    with sqlite3.connect(directory / "index.sqlite") as db, (directory / "lifetimes.bin").open("rb") as raw:
        query = ("select l.node,k.short_name,l.grid_x,l.block_x,l.warp_count,l.byte_offset,l.start_ns,l.chunk "
                 "from launches l join kernels k on k.id=l.kernel where l.phase=? and l.replay=?")
        args = [phase, replay]
        if node is not None:
            query += " and l.node=?"
            args.append(node)
        rows = db.execute(query + " order by l.node limit ?", [*args, max_nodes]).fetchall()
        require(rows, "replay or node absent")
        origin = min(r[6] for r in rows)
        tracks = {}
        for ordinal, name, grid, block, count, offset, _, chunk in rows:
            first_cta = 0 if cta is None else cta
            if first_cta >= grid:
                continue
            selected_count = min(max_ctas, grid) if cta is None else 1
            warp_bytes = ((block + 31) // 32) * RECORD.size
            raw.seek(offset + first_cta * warp_bytes)
            data = raw.read(selected_count * warp_bytes)
            for cta_id, warp, sm_id, start, end in RECORD.iter_unpack(data):
                if sm is not None and sm_id != sm:
                    continue
                if sm_id not in processes:
                    processes.add(sm_id)
                    events.append(dict(ph="M", name="process_name", pid=sm_id, tid=0, args={"name": f"SM {sm_id}"}))
                key = ordinal, cta_id, warp
                if key not in tracks:
                    tracks[key] = len(tracks) + 1
                    events.append(dict(ph="M", name="thread_name", pid=sm_id, tid=tracks[key],
                                       args={"name": f"launch {ordinal} CTA {cta_id} warp {warp}"}))
                details = dict(launch_ordinal=ordinal, replay=replay, phase=phase,
                               cta=cta_id, warp=warp, sm=sm_id, start_ns=start, end_ns=end, chunk=chunk)
                if phase == "graph" and name.startswith("days_"):
                    details["encoded_attempt"] = replay * 64 + ordinal // len(DAYS)
                events.append(dict(ph="X", name=name, pid=sm_id, tid=tracks[key], ts=(start-origin)/1000,
                                   dur=(end-start)/1000, args=details))
    Path(destination).write_text(json.dumps({"traceEvents": events, "displayTimeUnit": "ns"}, separators=(",", ":")))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    ingest_parser = commands.add_parser("ingest")
    for arg in ("directory", "simulator", "phase"):
        ingest_parser.add_argument(arg)
    for arg in ("first", "count"):
        ingest_parser.add_argument(arg, type=int)
    viewer = commands.add_parser("export")
    viewer.add_argument("directory")
    viewer.add_argument("destination")
    viewer.add_argument("--phase", choices=("graph","direct"), default="graph")
    viewer.add_argument("--replay", type=int, default=0)
    viewer.add_argument("--node", type=int)
    viewer.add_argument("--sm", type=int)
    viewer.add_argument("--cta", type=int)
    viewer.add_argument("--max-nodes", type=int, default=45)
    viewer.add_argument("--max-ctas", type=int, default=2)
    args = vars(parser.parse_args())
    command = args.pop("command")
    (ingest if command == "ingest" else export)(**args)
