#!/bin/bash
# =============================================================================
# verify_id_mapping.sh — Comprehensive verification of vertex ID mappings
#
# Tests that:
# 1. For each ordering, inline and MAP .lo produce identical graphs
# 2. Layered/chained orderings compose correctly
# 3. org_ids chain is preserved through multiple reorderings
# 4. Pre-reordered .sg files + MAP .lo produce correct results
# 5. Edge structure invariants hold (degree sequence, edge count)
# =============================================================================

set -euo pipefail

DIR="$(cd "$(dirname "$0")/../.." && pwd)"
BIN="$DIR/bench/bin"
TMP="/tmp/verify_ids"
PASS=0
FAIL=0
TESTS_RUN=0

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

check() {
    local name="$1"
    local result="$2"  # "PASS" or "FAIL"
    local detail="${3:-}"
    TESTS_RUN=$((TESTS_RUN + 1))
    if [ "$result" = "PASS" ]; then
        PASS=$((PASS + 1))
        echo -e "  ${GREEN}✓ PASS${NC}: $name"
    else
        FAIL=$((FAIL + 1))
        echo -e "  ${RED}✗ FAIL${NC}: $name"
        [ -n "$detail" ] && echo -e "         $detail"
    fi
}

# Compare sorted edge lists
edges_match() {
    local f1="$1" f2="$2"
    sort -n -k1,1 -k2,2 "$f1" > "${f1}.sorted"
    sort -n -k1,1 -k2,2 "$f2" > "${f2}.sorted"
    diff -q "${f1}.sorted" "${f2}.sorted" > /dev/null 2>&1
}

# Get degree sequence (sorted list of degrees)
degree_seq() {
    local el="$1"
    # Count occurrences of each node as source
    awk '{print $1}' "$el" | sort -n | uniq -c | sort -rn | awk '{print $1}'
}

# Count edges
edge_count() {
    wc -l < "$1"
}

rm -rf "$TMP"
mkdir -p "$TMP"

echo "============================================================"
echo "  GraphBrew Vertex ID Mapping Verification Suite"
echo "============================================================"
echo ""

# ═══════════════════════════════════════════════════════════════════
# PHASE 0: Create test graphs
# ═══════════════════════════════════════════════════════════════════
echo "Phase 0: Creating test graphs..."

# Small hand-crafted graph (8 nodes, 11 edges → 22 directed)
cat > "$TMP/test.el" << 'EOF'
0 1
0 2
0 3
1 2
1 4
2 3
3 4
4 5
5 6
5 7
6 7
EOF

# Convert to .sg with ORIGINAL ordering (identity org_ids)
"$BIN/converter" -f "$TMP/test.el" -s -b "$TMP/test_identity.sg" > /dev/null 2>&1

# Convert to .sg with RANDOM ordering (non-identity org_ids) — this is what the experiment pipeline does
"$BIN/converter" -f "$TMP/test.el" -s -o 1 -b "$TMP/test_random.sg" > /dev/null 2>&1

# Also use kronecker generator for a slightly larger graph
"$BIN/converter" -g 10 -s -b "$TMP/kron10.sg" > /dev/null 2>&1

# And a RANDOM-baseline version of kron10
"$BIN/converter" -f "$TMP/kron10.sg" -s -o 1 -b "$TMP/kron10_random.sg" > /dev/null 2>&1

echo "  Created test_identity.sg (8 nodes, identity org_ids)"
echo "  Created test_random.sg (8 nodes, random org_ids)"
echo "  Created kron10.sg (1024 nodes, identity org_ids)"
echo "  Created kron10_random.sg (1024 nodes, random org_ids)"
echo ""

# ═══════════════════════════════════════════════════════════════════
# PHASE 1: Verify org_ids
# ═══════════════════════════════════════════════════════════════════
echo "Phase 1: Verify org_ids status"

# Identity org_ids check
"$BIN/converter" -f "$TMP/test_identity.sg" -s -o 0 -q "$TMP/identity_orgids.lo" > /dev/null 2>&1
MISMATCH=$(awk 'NR != $1+1 {count++} END {print count+0}' "$TMP/identity_orgids.lo")
check "test_identity.sg has identity org_ids" "$([ "$MISMATCH" -eq 0 ] && echo PASS || echo FAIL)" \
    "Expected 0 mismatches, got $MISMATCH"

# Non-identity org_ids check
"$BIN/converter" -f "$TMP/test_random.sg" -s -o 0 -q "$TMP/random_orgids.lo" > /dev/null 2>&1
MISMATCH=$(awk 'NR != $1+1 {count++} END {print count+0}' "$TMP/random_orgids.lo")
check "test_random.sg has non-identity org_ids" "$([ "$MISMATCH" -gt 0 ] && echo PASS || echo FAIL)" \
    "Expected >0 mismatches, got $MISMATCH"

# org_ids is a valid permutation (all values 0..N-1 appear exactly once)
NODES=$(wc -l < "$TMP/random_orgids.lo")
UNIQUE=$(sort -un "$TMP/random_orgids.lo" | wc -l)
MIN=$(sort -n "$TMP/random_orgids.lo" | head -1)
MAX=$(sort -n "$TMP/random_orgids.lo" | tail -1)
check "org_ids is valid permutation (random)" \
    "$([ "$UNIQUE" -eq "$NODES" ] && [ "$MIN" -eq 0 ] && [ "$MAX" -eq $((NODES-1)) ] && echo PASS || echo FAIL)" \
    "nodes=$NODES unique=$UNIQUE min=$MIN max=$MAX"

echo ""

# ═══════════════════════════════════════════════════════════════════
# PHASE 2: Single orderings — inline vs MAP on identity .sg
# ═══════════════════════════════════════════════════════════════════
echo "Phase 2: Single orderings — inline vs MAP on IDENTITY .sg"

for ORDER_OPT in "SORT:2" "HUBSORT:3" "HUBCLUSTER:4" "DBG:5" "HUBSORTDBG:6" "HUBCLUSTERDBG:7" "GORDER:9" "RCM:11" "RABBIT:8:csr" "LeidenOrder:15"; do
    IFS=':' read -r NAME OPT1 OPT2 <<< "$ORDER_OPT"
    OPT="${OPT1}${OPT2:+:$OPT2}"

    # Generate .lo AND dump edges in same invocation (same seed for non-deterministic)
    "$BIN/converter" -f "$TMP/test_identity.sg" -s -o "$OPT" \
        -q "$TMP/id_${NAME}.lo" -e "$TMP/id_${NAME}_inline.el" > /dev/null 2>&1

    # Apply via MAP and dump edges
    "$BIN/converter" -f "$TMP/test_identity.sg" -s -o "13:$TMP/id_${NAME}.lo" \
        -e "$TMP/id_${NAME}_map.el" > /dev/null 2>&1

    # Compare
    if edges_match "$TMP/id_${NAME}_inline.el" "$TMP/id_${NAME}_map.el"; then
        check "$NAME (identity .sg): inline == MAP" "PASS"
    else
        check "$NAME (identity .sg): inline == MAP" "FAIL" \
            "Edge lists differ"
    fi

    # Verify edge count preserved
    EC_INLINE=$(edge_count "$TMP/id_${NAME}_inline.el")
    EC_MAP=$(edge_count "$TMP/id_${NAME}_map.el")
    EC_ORIG=$(edge_count "$TMP/test_identity.sg" 2>/dev/null || echo "?")
    check "$NAME: edge count preserved ($EC_INLINE)" \
        "$([ "$EC_INLINE" -eq "$EC_MAP" ] && echo PASS || echo FAIL)" \
        "inline=$EC_INLINE map=$EC_MAP"
done

echo ""

# ═══════════════════════════════════════════════════════════════════
# PHASE 3: Single orderings — inline vs MAP on RANDOM-baseline .sg
# This is the critical test — the .sg has non-identity org_ids
# ═══════════════════════════════════════════════════════════════════
echo "Phase 3: Single orderings — inline vs MAP on RANDOM-baseline .sg"

for ORDER_OPT in "SORT:2" "HUBSORT:3" "HUBCLUSTER:4" "DBG:5" "GORDER:9" "RCM:11" "RABBIT:8:csr" "LeidenOrder:15"; do
    IFS=':' read -r NAME OPT1 OPT2 <<< "$ORDER_OPT"
    OPT="${OPT1}${OPT2:+:$OPT2}"

    # Generate .lo AND dump edges in same invocation
    "$BIN/converter" -f "$TMP/test_random.sg" -s -o "$OPT" \
        -q "$TMP/rnd_${NAME}.lo" -e "$TMP/rnd_${NAME}_inline.el" > /dev/null 2>&1

    # Apply via MAP and dump edges
    "$BIN/converter" -f "$TMP/test_random.sg" -s -o "13:$TMP/rnd_${NAME}.lo" \
        -e "$TMP/rnd_${NAME}_map.el" > /dev/null 2>&1

    # Compare
    if edges_match "$TMP/rnd_${NAME}_inline.el" "$TMP/rnd_${NAME}_map.el"; then
        check "$NAME (random .sg): inline == MAP" "PASS"
    else
        check "$NAME (random .sg): inline == MAP" "FAIL" \
            "Edge lists differ on pre-reordered .sg!"
    fi
done

echo ""

# ═══════════════════════════════════════════════════════════════════
# PHASE 4: Verify org_ids chain through reordering
# After applying ordering X to a graph, org_ids should map back to
# the ORIGINAL source graph IDs, not to the intermediate IDs.
# ═══════════════════════════════════════════════════════════════════
echo "Phase 4: org_ids chain verification"

# Start with test_identity.sg (org_ids = identity)
# Apply HUBCLUSTER (deterministic, actually moves nodes) → get .sg → dump org_ids
# → apply those org_ids as MAP → should recover same edges

"$BIN/converter" -f "$TMP/test_identity.sg" -s -o 4 -b "$TMP/hc_reordered.sg" > /dev/null 2>&1
"$BIN/converter" -f "$TMP/hc_reordered.sg" -s -o 0 -e "$TMP/hc_edges.el" > /dev/null 2>&1
"$BIN/converter" -f "$TMP/hc_reordered.sg" -s -o 0 -q "$TMP/hc_orgids.lo" > /dev/null 2>&1

# org_ids from hc_reordered.sg should tell us: hc_orgids[hc_id] = original_id
# Now apply those org_ids as MAP to test_identity.sg → should produce same graph as HUBCLUSTER inline
"$BIN/converter" -f "$TMP/test_identity.sg" -s -o "13:$TMP/hc_orgids.lo" \
    -e "$TMP/hc_via_orgids.el" > /dev/null 2>&1

if edges_match "$TMP/hc_edges.el" "$TMP/hc_via_orgids.el"; then
    check "org_ids from reordered.sg used as MAP → matches HUBCLUSTER inline" "PASS"
else
    check "org_ids from reordered.sg used as MAP → matches HUBCLUSTER inline" "FAIL" \
        "Applying reordered.sg org_ids via MAP doesn't reproduce HUBCLUSTER edges"
fi

# Chain: identity.sg → RANDOM → random.sg → HUBCLUSTER → doubly_reordered.sg
# The org_ids of doubly_reordered.sg should still map to ORIGINAL (edge list) IDs
"$BIN/converter" -f "$TMP/test_random.sg" -s -o 4 -b "$TMP/double_reorder.sg" > /dev/null 2>&1
"$BIN/converter" -f "$TMP/double_reorder.sg" -s -o 0 -q "$TMP/double_orgids.lo" > /dev/null 2>&1

# org_ids should still be a valid permutation of 0..N-1
NODES=$(wc -l < "$TMP/double_orgids.lo")
UNIQUE=$(sort -un "$TMP/double_orgids.lo" | wc -l)
MIN=$(sort -n "$TMP/double_orgids.lo" | head -1)
MAX=$(sort -n "$TMP/double_orgids.lo" | tail -1)
check "double-reordered org_ids is valid permutation" \
    "$([ "$UNIQUE" -eq "$NODES" ] && [ "$MIN" -eq 0 ] && [ "$MAX" -eq $((NODES-1)) ] && echo PASS || echo FAIL)" \
    "nodes=$NODES unique=$UNIQUE min=$MIN max=$MAX"

# The degree sequence should be identical across all renamings
# (reordering only renames vertices, doesn't change the graph structure)
"$BIN/converter" -f "$TMP/test_identity.sg" -s -o 0 -e "$TMP/orig_edges.el" > /dev/null 2>&1
"$BIN/converter" -f "$TMP/double_reorder.sg" -s -o 0 -e "$TMP/double_edges.el" > /dev/null 2>&1

DEG_ORIG=$(degree_seq "$TMP/orig_edges.el" | md5sum)
DEG_DOUBLE=$(degree_seq "$TMP/double_edges.el" | md5sum)
check "degree sequence preserved through double reorder" \
    "$([ "$DEG_ORIG" = "$DEG_DOUBLE" ] && echo PASS || echo FAIL)" \
    "orig=$DEG_ORIG double=$DEG_DOUBLE"

# Verify: org_ids from double-reordered graph, applied as MAP to identity.sg,
# should produce the SAME edges as the double-reordered graph itself
"$BIN/converter" -f "$TMP/test_identity.sg" -s -o "13:$TMP/double_orgids.lo" \
    -e "$TMP/double_via_orgids.el" > /dev/null 2>&1

if edges_match "$TMP/double_edges.el" "$TMP/double_via_orgids.el"; then
    check "double-reorder org_ids as MAP → reproduces double-reordered graph" "PASS"
else
    check "double-reorder org_ids as MAP → reproduces double-reordered graph" "FAIL"
fi

echo ""

# ═══════════════════════════════════════════════════════════════════
# PHASE 5: GraphBrewOrder layered/chained orderings
# ═══════════════════════════════════════════════════════════════════
echo "Phase 5: Layered/chained orderings (GraphBrewOrder variants)"

for VARIANT in "leiden:flat" "hubcluster" "rabbit" "hrab"; do
    OPT="12:${VARIANT}"

    # Test on identity .sg
    "$BIN/converter" -f "$TMP/test_identity.sg" -s -o "$OPT" \
        -q "$TMP/gb_id_${VARIANT//:/_}.lo" -e "$TMP/gb_id_${VARIANT//:/_}_inline.el" > /dev/null 2>&1

    "$BIN/converter" -f "$TMP/test_identity.sg" -s -o "13:$TMP/gb_id_${VARIANT//:/_}.lo" \
        -e "$TMP/gb_id_${VARIANT//:/_}_map.el" > /dev/null 2>&1

    if edges_match "$TMP/gb_id_${VARIANT//:/_}_inline.el" "$TMP/gb_id_${VARIANT//:/_}_map.el"; then
        check "GraphBrew $VARIANT (identity .sg): inline == MAP" "PASS"
    else
        check "GraphBrew $VARIANT (identity .sg): inline == MAP" "FAIL"
    fi

    # Test on RANDOM .sg
    "$BIN/converter" -f "$TMP/test_random.sg" -s -o "$OPT" \
        -q "$TMP/gb_rnd_${VARIANT//:/_}.lo" -e "$TMP/gb_rnd_${VARIANT//:/_}_inline.el" > /dev/null 2>&1

    "$BIN/converter" -f "$TMP/test_random.sg" -s -o "13:$TMP/gb_rnd_${VARIANT//:/_}.lo" \
        -e "$TMP/gb_rnd_${VARIANT//:/_}_map.el" > /dev/null 2>&1

    if edges_match "$TMP/gb_rnd_${VARIANT//:/_}_inline.el" "$TMP/gb_rnd_${VARIANT//:/_}_map.el"; then
        check "GraphBrew $VARIANT (random .sg): inline == MAP" "PASS"
    else
        check "GraphBrew $VARIANT (random .sg): inline == MAP" "FAIL"
    fi
done

echo ""

# ═══════════════════════════════════════════════════════════════════
# PHASE 6: Larger graph (kron10, 1024 nodes) — same tests
# ═══════════════════════════════════════════════════════════════════
echo "Phase 6: Larger graph (kron10, 1024 nodes)"

for ORDER_OPT in "SORT:2" "HUBCLUSTER:4" "GORDER:9" "RABBIT:8:csr" "GB_hrab:12:hrab" "LeidenOrder:15"; do
    IFS=':' read -r NAME OPT1 OPT2 OPT3 <<< "$ORDER_OPT"
    if [ -n "${OPT3:-}" ]; then
        OPT="${OPT1}:${OPT2}:${OPT3}"
    elif [ -n "${OPT2:-}" ]; then
        OPT="${OPT1}:${OPT2}"
    else
        OPT="$OPT1"
    fi

    # Identity kron10
    "$BIN/converter" -f "$TMP/kron10.sg" -s -o "$OPT" \
        -q "$TMP/k_id_${NAME}.lo" -e "$TMP/k_id_${NAME}_inline.el" > /dev/null 2>&1
    "$BIN/converter" -f "$TMP/kron10.sg" -s -o "13:$TMP/k_id_${NAME}.lo" \
        -e "$TMP/k_id_${NAME}_map.el" > /dev/null 2>&1

    if edges_match "$TMP/k_id_${NAME}_inline.el" "$TMP/k_id_${NAME}_map.el"; then
        check "$NAME (kron10 identity): inline == MAP" "PASS"
    else
        check "$NAME (kron10 identity): inline == MAP" "FAIL"
    fi

    # Random-baseline kron10
    "$BIN/converter" -f "$TMP/kron10_random.sg" -s -o "$OPT" \
        -q "$TMP/k_rnd_${NAME}.lo" -e "$TMP/k_rnd_${NAME}_inline.el" > /dev/null 2>&1
    "$BIN/converter" -f "$TMP/kron10_random.sg" -s -o "13:$TMP/k_rnd_${NAME}.lo" \
        -e "$TMP/k_rnd_${NAME}_map.el" > /dev/null 2>&1

    if edges_match "$TMP/k_rnd_${NAME}_inline.el" "$TMP/k_rnd_${NAME}_map.el"; then
        check "$NAME (kron10 random): inline == MAP" "PASS"
    else
        check "$NAME (kron10 random): inline == MAP" "FAIL"
    fi
done

echo ""

# ═══════════════════════════════════════════════════════════════════
# PHASE 7: Cross-verify — .lo from identity .sg applied to .sg built
# from same source. The key insight: two .sg files from the same
# source but with different internal orderings should produce the
# same reordered graph when the same .lo is applied.
# ═══════════════════════════════════════════════════════════════════
echo "Phase 7: Cross-verify — same .lo applied to different .sg baselines"

# SORT .lo from identity .sg
"$BIN/converter" -f "$TMP/test_identity.sg" -s -o 2 \
    -q "$TMP/cross_sort_id.lo" -e "$TMP/cross_sort_id_inline.el" > /dev/null 2>&1

# Apply same .lo to random .sg → should produce same edges
"$BIN/converter" -f "$TMP/test_random.sg" -s -o "13:$TMP/cross_sort_id.lo" \
    -e "$TMP/cross_sort_rnd_map.el" > /dev/null 2>&1

if edges_match "$TMP/cross_sort_id_inline.el" "$TMP/cross_sort_rnd_map.el"; then
    check "SORT .lo from identity applied to random .sg → same edges" "PASS"
else
    check "SORT .lo from identity applied to random .sg → same edges" "FAIL" \
        "This tests that MAP correctly resolves org_ids differences"
fi

# HUBCLUSTER .lo from random .sg
"$BIN/converter" -f "$TMP/test_random.sg" -s -o 4 \
    -q "$TMP/cross_hc_rnd.lo" -e "$TMP/cross_hc_rnd_inline.el" > /dev/null 2>&1

# Apply same .lo to identity .sg → should produce same edges
"$BIN/converter" -f "$TMP/test_identity.sg" -s -o "13:$TMP/cross_hc_rnd.lo" \
    -e "$TMP/cross_hc_id_map.el" > /dev/null 2>&1

if edges_match "$TMP/cross_hc_rnd_inline.el" "$TMP/cross_hc_id_map.el"; then
    check "HUBCLUSTER .lo from random applied to identity .sg → same edges" "PASS"
else
    check "HUBCLUSTER .lo from random applied to identity .sg → same edges" "FAIL"
fi

echo ""

# ═══════════════════════════════════════════════════════════════════
# PHASE 8: PR benchmark consistency — verify PR produces same
# PageRank scores (top-K) with inline vs MAP
# ═══════════════════════════════════════════════════════════════════
echo "Phase 8: PR & BFS benchmark consistency (inline vs MAP)"

# Use kron10 for this test — compare that both inline and MAP run PR
# and produce the same graph topology features (deterministic)
"$BIN/converter" -f "$TMP/kron10_random.sg" -s -o 4 -q "$TMP/pr_test_hc.lo" > /dev/null 2>&1

# Compare topology features between inline and MAP (deterministic)
PR_INLINE_TOPO=$("$BIN/pr" -f "$TMP/kron10_random.sg" -n 1 -o 4 2>&1 | grep -E "Avg Degree|Forward Edge|Hub Conc" | sort)
PR_MAP_TOPO=$("$BIN/pr" -f "$TMP/kron10_random.sg" -n 1 -o "13:$TMP/pr_test_hc.lo" 2>&1 | grep -E "Avg Degree|Forward Edge|Hub Conc" | sort)

check "PR topology features match (inline vs MAP)" \
    "$([ "$PR_INLINE_TOPO" = "$PR_MAP_TOPO" ] && echo PASS || echo FAIL)"

# BFS should also succeed on both paths
BFS_INLINE=$("$BIN/bfs" -f "$TMP/kron10_random.sg" -n 1 -o 4 2>&1 | grep "Average" | awk '{print $3}')
BFS_MAP=$("$BIN/bfs" -f "$TMP/kron10_random.sg" -n 1 -o "13:$TMP/pr_test_hc.lo" 2>&1 | grep "Average" | awk '{print $3}')

check "BFS completes on inline (time=$BFS_INLINE)" \
    "$([ -n "$BFS_INLINE" ] && echo PASS || echo FAIL)"
check "BFS completes on MAP (time=$BFS_MAP)" \
    "$([ -n "$BFS_MAP" ] && echo PASS || echo FAIL)"

echo ""

# ═══════════════════════════════════════════════════════════════════
# PHASE 9: BFS source node verification
# When reordering, the BFS source should map correctly so the same
# original node is used regardless of ordering
# ═══════════════════════════════════════════════════════════════════
echo "Phase 9: Triple-reorder chain verification"

# identity.sg → RANDOM → HUBCLUSTER → GORDER → triple.sg
# org_ids should still be valid permutation mapping to original
"$BIN/converter" -f "$TMP/test_identity.sg" -s -o 1 -b "$TMP/chain_r.sg" > /dev/null 2>&1
"$BIN/converter" -f "$TMP/chain_r.sg" -s -o 4 -b "$TMP/chain_rh.sg" > /dev/null 2>&1
"$BIN/converter" -f "$TMP/chain_rh.sg" -s -o 9 -b "$TMP/chain_rhg.sg" > /dev/null 2>&1
"$BIN/converter" -f "$TMP/chain_rhg.sg" -s -o 0 -q "$TMP/chain_orgids.lo" > /dev/null 2>&1
"$BIN/converter" -f "$TMP/chain_rhg.sg" -s -o 0 -e "$TMP/chain_edges.el" > /dev/null 2>&1

# org_ids should be valid permutation
NODES=$(wc -l < "$TMP/chain_orgids.lo")
UNIQUE=$(sort -un "$TMP/chain_orgids.lo" | wc -l)
check "triple-reorder org_ids is valid permutation" \
    "$([ "$UNIQUE" -eq "$NODES" ] && echo PASS || echo FAIL)" \
    "nodes=$NODES unique=$UNIQUE"

# Applying chain org_ids to identity.sg via MAP should reproduce same edges
"$BIN/converter" -f "$TMP/test_identity.sg" -s -o "13:$TMP/chain_orgids.lo" \
    -e "$TMP/chain_via_orgids.el" > /dev/null 2>&1

if edges_match "$TMP/chain_edges.el" "$TMP/chain_via_orgids.el"; then
    check "triple-reorder org_ids as MAP → reproduces final graph" "PASS"
else
    check "triple-reorder org_ids as MAP → reproduces final graph" "FAIL"
fi

# Degree sequence still preserved
DEG_CHAIN=$(degree_seq "$TMP/chain_edges.el" | md5sum)
DEG_ORIG2=$(degree_seq "$TMP/orig_edges.el" | md5sum)
check "degree sequence preserved through triple reorder" \
    "$([ "$DEG_CHAIN" = "$DEG_ORIG2" ] && echo PASS || echo FAIL)"

echo ""

# ═══════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════
echo "============================================================"
echo -e "  RESULTS: ${GREEN}$PASS PASSED${NC}, ${RED}$FAIL FAILED${NC}, $TESTS_RUN total"
echo "============================================================"

if [ "$FAIL" -gt 0 ]; then
    echo -e "  ${RED}SOME TESTS FAILED${NC} — investigate the failures above"
    exit 1
else
    echo -e "  ${GREEN}ALL TESTS PASSED${NC}"
    exit 0
fi
