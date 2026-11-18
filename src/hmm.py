"""
PathSnap — HMM + Viterbi Decoder
src/hmm.py

Takes per-fix emission probabilities (from ML model) and
road network transition probabilities to decode the most
likely arc sequence for a trip.
"""

import numpy as np
import networkx as nx


# ─────────────────────────────────────────────
#  Road graph builder
# ─────────────────────────────────────────────

def build_road_graph(arcs_df):
    """
    Build a directed NetworkX graph from the arcs DataFrame.
    Nodes = node_ids, Edges = directed arcs with arc_id attribute.
    """
    G = nx.DiGraph()
    for _, row in arcs_df.iterrows():
        G.add_edge(
            int(row['from_node_id']),
            int(row['to_node_id']),
            arc_id=int(row['arc_id'])
        )
    return G


def arc_to_nodes(arcs_df):
    """Return dict: arc_id → (from_node_id, to_node_id)."""
    return {
        int(row['arc_id']): (int(row['from_node_id']), int(row['to_node_id']))
        for _, row in arcs_df.iterrows()
    }


# ─────────────────────────────────────────────
#  Transition probability
# ─────────────────────────────────────────────

def get_transition_prob(from_arc_id, to_arc_id, arc_to_nodes_map, graph,
                        connected_prob=0.9, unconnected_prob=1e-6):
    """
    P(to_arc | from_arc) based on road connectivity.

    If the to_node of from_arc equals the from_node of to_arc
    (i.e. they are directly connected), return connected_prob / out_degree.
    Otherwise return a small near-zero probability.
    """
    if from_arc_id not in arc_to_nodes_map or to_arc_id not in arc_to_nodes_map:
        return unconnected_prob

    _, from_arc_to_node = arc_to_nodes_map[from_arc_id]
    to_arc_from_node, _ = arc_to_nodes_map[to_arc_id]

    if from_arc_to_node == to_arc_from_node:
        # Directly connected — spread probability equally over out-arcs
        out_degree = graph.out_degree(from_arc_to_node)
        if out_degree == 0:
            return unconnected_prob
        return connected_prob / out_degree

    # Check if reachable within 2 hops (u-turns, short detours)
    try:
        path_len = nx.shortest_path_length(graph, from_arc_to_node, to_arc_from_node)
        if path_len <= 2:
            return (connected_prob * 0.3) / max(1, graph.out_degree(from_arc_to_node))
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        pass

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
        arc_to_nodes_map       : dict arc_id → (from_node, to_node)
        graph                  : NetworkX DiGraph
        gap_indices            : set of fix indices where the chain should RESET
                                 (detected GPS gaps > threshold)

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
        return math.log(p) if p > 0 else LOG_ZERO

    import math

    # viterbi[t][arc_id] = log probability of best path ending at arc_id at fix t
    viterbi = [{} for _ in range(n_fixes)]
    backtrack = [{} for _ in range(n_fixes)]

    # --- Initialisation (fix 0) ---
    candidates_0 = candidate_arcs_per_fix[0]
    emits_0 = emission_probs_per_fix[0]
    n_cands = max(len(candidates_0), 1)
    for arc_id in candidates_0:
        emit = emits_0.get(arc_id, 1e-9)
        # Uniform initial distribution
        viterbi[0][arc_id] = safe_log(1.0 / n_cands) + safe_log(emit)
        backtrack[0][arc_id] = None

    # --- Recursion ---
    for t in range(1, n_fixes):
        candidates_t = candidate_arcs_per_fix[t]
        emits_t = emission_probs_per_fix[t]
        candidates_prev = candidate_arcs_per_fix[t - 1]
        reset = t in gap_indices  # GPS gap → reset chain

        for arc_id in candidates_t:
            emit = emits_t.get(arc_id, 1e-9)
            log_emit = safe_log(emit)

            if reset:
                # Start fresh: ignore previous state
                n_cands_t = max(len(candidates_t), 1)
                viterbi[t][arc_id] = safe_log(1.0 / n_cands_t) + log_emit
                backtrack[t][arc_id] = None
            else:
                best_log_prob = LOG_ZERO
                best_prev = None
                for prev_arc in candidates_prev:
                    if prev_arc not in viterbi[t - 1]:
                        continue
                    trans = get_transition_prob(prev_arc, arc_id,
                                               arc_to_nodes_map, graph)
                    log_prob = viterbi[t - 1][prev_arc] + safe_log(trans) + log_emit
                    if log_prob > best_log_prob:
                        best_log_prob = log_prob
                        best_prev = prev_arc

                viterbi[t][arc_id] = best_log_prob
                backtrack[t][arc_id] = best_prev

    # --- Termination: best arc at last fix ---
    if not viterbi[n_fixes - 1]:
        return [None] * n_fixes

    best_last = max(viterbi[n_fixes - 1], key=viterbi[n_fixes - 1].get)

    # --- Backtrack ---
    path = [None] * n_fixes
    path[n_fixes - 1] = best_last
    for t in range(n_fixes - 2, -1, -1):
        path[t] = backtrack[t + 1].get(path[t + 1])

    return path


# ─────────────────────────────────────────────
#  Gap detection
# ─────────────────────────────────────────────

def detect_gaps(track_df, gap_threshold_sec=5.0):
    """
    Return set of fix indices where time gap from previous fix exceeds threshold.
    These are reset points for the Viterbi chain.
    """
    gaps = set()
    timestamps = track_df['timestamp_sec'].values
    for i in range(1, len(timestamps)):
        if timestamps[i] - timestamps[i - 1] > gap_threshold_sec:
            gaps.add(i)
    return gaps


# ─────────────────────────────────────────────
#  Metrics
# ─────────────────────────────────────────────

def route_continuity(predicted_arcs, arc_to_nodes_map):
    """
    Fraction of consecutive predicted arc pairs that are directly connected.
    Skips None predictions.
    """
    valid_pairs = [(a, b) for a, b in zip(predicted_arcs, predicted_arcs[1:])
                   if a is not None and b is not None]
    if not valid_pairs:
        return 0.0

    connected = 0
    for a, b in valid_pairs:
        if a in arc_to_nodes_map and b in arc_to_nodes_map:
            _, a_to = arc_to_nodes_map[a]
            b_from, _ = arc_to_nodes_map[b]
            if a_to == b_from:
                connected += 1
    return connected / len(valid_pairs)
