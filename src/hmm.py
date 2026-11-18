"""
PathSnap — HMM + Viterbi Decoder
src/hmm.py

Takes per-fix emission probabilities (from ML model) and
road network transition probabilities to decode the most
likely arc sequence for a trip.
"""

import math
import numpy as np
import networkx as nx


# ─────────────────────────────────────────────
#  Road graph builder (fast vectorized)
# ─────────────────────────────────────────────

def build_road_graph(arcs_df):
    """
    Build a directed NetworkX graph from the arcs DataFrame.
    Uses fast zip over numpy arrays (100x faster than iterrows).
    """
    G = nx.DiGraph()
    from_ids = arcs_df['from_node_id'].to_numpy(dtype=np.int64).tolist()
    to_ids   = arcs_df['to_node_id'].to_numpy(dtype=np.int64).tolist()
    arc_ids  = arcs_df['arc_id'].to_numpy(dtype=np.int64).tolist()
    G.add_edges_from(
        (u, v, {'arc_id': a}) for u, v, a in zip(from_ids, to_ids, arc_ids)
    )
    return G


def arc_to_nodes(arcs_df):
    """Return dict: arc_id -> (from_node_id, to_node_id)."""
    arc_ids  = arcs_df['arc_id'].to_numpy(dtype=np.int64).tolist()
    from_ids = arcs_df['from_node_id'].to_numpy(dtype=np.int64).tolist()
    to_ids   = arcs_df['to_node_id'].to_numpy(dtype=np.int64).tolist()
    return dict(zip(arc_ids, zip(from_ids, to_ids)))


# ─────────────────────────────────────────────
#  Transition probability (fast O(deg) lookup)
# ─────────────────────────────────────────────

def get_transition_prob(from_arc_id, to_arc_id, arc_to_nodes_map, graph,
                        same_arc_prob=0.85, connected_prob=0.9,
                        two_hop_prob=0.25, unconnected_prob=1e-6):
    """
    P(to_arc | from_arc) based on road connectivity.
      - Staying on the same arc (common at 1s GPS rate): same_arc_prob
      - Direct 1-hop transition (to_node == from_node): connected_prob / out_degree
      - 2-hop transition (short intermediate arc skipped between fixes): two_hop_prob / out_degree
      - Otherwise: unconnected_prob
    """
    if from_arc_id == to_arc_id:
        return same_arc_prob

    if from_arc_id not in arc_to_nodes_map or to_arc_id not in arc_to_nodes_map:
        return unconnected_prob

    _, from_to = arc_to_nodes_map[from_arc_id]
    to_from, _ = arc_to_nodes_map[to_arc_id]

    if from_to == to_from:
        out_deg = graph.out_degree(from_to)
        return connected_prob / max(1, out_deg)

    # Fast 1-edge / 2-edge intermediate check (avoids expensive full-graph BFS)
    adj = graph._succ
    if from_to in adj:
        nbrs_1 = adj[from_to]
        if to_from in nbrs_1:
            return two_hop_prob / max(1, len(nbrs_1))
        for mid in nbrs_1:
            if mid in adj and to_from in adj[mid]:
                return (two_hop_prob * 0.4) / max(1, len(nbrs_1))

    return unconnected_prob


# ─────────────────────────────────────────────
#  Viterbi decoder
# ─────────────────────────────────────────────

def viterbi_decode(emission_probs_per_fix, candidate_arcs_per_fix,
                   arc_to_nodes_map, graph,
                   gap_indices=None):
    """
    Viterbi algorithm for HMM-based map matching.

    Args:
        emission_probs_per_fix : list of dicts {arc_id: P(obs|arc)} per GPS fix
        candidate_arcs_per_fix : list of lists of arc_ids per GPS fix
        arc_to_nodes_map       : dict arc_id -> (from_node, to_node)
        graph                  : NetworkX DiGraph
        gap_indices            : set of fix indices where the chain resets (GPS gaps)

    Returns:
        List of predicted arc_ids, one per GPS fix.
    """
    if gap_indices is None:
        gap_indices = set()

    n_fixes = len(emission_probs_per_fix)
    if n_fixes == 0:
        return []

    LOG_ZERO = -1e18

    def safe_log(p):
        return math.log(p) if p > 1e-12 else math.log(1e-12)

    viterbi = [{} for _ in range(n_fixes)]
    backtrack = [{} for _ in range(n_fixes)]

    # --- Initialisation (fix 0) ---
    candidates_0 = candidate_arcs_per_fix[0]
    emits_0 = emission_probs_per_fix[0]
    n_cands = max(len(candidates_0), 1)
    for arc_id in candidates_0:
        emit = emits_0.get(arc_id, 1e-9)
        viterbi[0][arc_id] = safe_log(1.0 / n_cands) + safe_log(emit)
        backtrack[0][arc_id] = None

    # --- Recursion ---
    for t in range(1, n_fixes):
        candidates_t = candidate_arcs_per_fix[t]
        emits_t = emission_probs_per_fix[t]
        candidates_prev = candidate_arcs_per_fix[t - 1]
        reset = (t in gap_indices) or (not viterbi[t - 1])

        if not candidates_t:
            continue

        n_cands_t = max(len(candidates_t), 1)
        for arc_id in candidates_t:
            emit = emits_t.get(arc_id, 1e-9)
            log_emit = safe_log(emit)

            if reset:
                viterbi[t][arc_id] = safe_log(1.0 / n_cands_t) + log_emit
                backtrack[t][arc_id] = None
            else:
                best_log_prob = LOG_ZERO
                best_prev = None
                for prev_arc in candidates_prev:
                    if prev_arc not in viterbi[t - 1]:
                        continue
                    trans = get_transition_prob(
                        prev_arc, arc_id, arc_to_nodes_map, graph
                    )
                    log_prob = viterbi[t - 1][prev_arc] + safe_log(trans) + log_emit
                    if log_prob > best_log_prob:
                        best_log_prob = log_prob
                        best_prev = prev_arc

                viterbi[t][arc_id] = best_log_prob
                backtrack[t][arc_id] = best_prev

    # --- Backtrack (handles GPS gap resets cleanly without wiping earlier segments) ---
    path = [None] * n_fixes
    for t in range(n_fixes - 1, -1, -1):
        if not viterbi[t]:
            continue
        if t == n_fixes - 1 or path[t + 1] is None:
            path[t] = max(viterbi[t], key=viterbi[t].get)
        else:
            prev = backtrack[t + 1].get(path[t + 1])
            if prev is not None and prev in viterbi[t]:
                path[t] = prev
            else:
                # Reset boundary (GPS gap): pick best ending state of previous segment
                path[t] = max(viterbi[t], key=viterbi[t].get)

    return path


# ─────────────────────────────────────────────
#  Gap detection
# ─────────────────────────────────────────────

def detect_gaps(track_df, gap_threshold_sec=5.0):
    """Return set of fix indices where time gap exceeds threshold."""
    gaps = set()
    timestamps = track_df['timestamp_sec'].to_numpy()
    for i in range(1, len(timestamps)):
        if timestamps[i] - timestamps[i - 1] > gap_threshold_sec:
            gaps.add(i)
    return gaps


# ─────────────────────────────────────────────
#  Continuity Metric
# ─────────────────────────────────────────────

def route_continuity(predicted_arcs, arc_to_nodes_map):
    """
    Fraction of consecutive predicted arc pairs that are either:
      - on the same arc (a == b), OR
      - directly connected (to_node(a) == from_node(b)).
    """
    valid_pairs = [(a, b) for a, b in zip(predicted_arcs, predicted_arcs[1:])
                   if a is not None and b is not None]
    if not valid_pairs:
        return 0.0

    connected = 0
    for a, b in valid_pairs:
        if a == b:
            connected += 1
        elif a in arc_to_nodes_map and b in arc_to_nodes_map:
            _, a_to = arc_to_nodes_map[a]
            b_from, _ = arc_to_nodes_map[b]
            if a_to == b_from:
                connected += 1
    return connected / len(valid_pairs)
