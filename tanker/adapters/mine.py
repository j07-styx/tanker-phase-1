"""Tanker Run (CVRP) solver.

Clarke-Wright savings start, then SISR-style ruin & recreate (Christiaens &
Vanden Berghe 2020): each iteration removes a few strings of consecutive villages
from neighbouring routes and greedily reinserts them with "blinks", accepted by
simulated annealing. Every plan stays capacity- and fleet-feasible.
"""

import math
import random
import time

from adapter import Solver
from adapters.starter import StarterSolver
from data import distance_matrix

STOP_AT_S = 5.55       # stop searching this long after solve() starts (budget 6.0 s)
SUBMIT_EVERY_S = 0.12  # throttle for submitting improvements
CBAR = 10              # average villages removed per iteration
LMAX = 10              # longest string removed from one route
BLINK = 0.01           # chance an insertion position is skipped
T0_FACTOR = 0.7        # start temperature, in average legs
T1_FACTOR = 0.01       # end temperature
NB = 16                # insertion positions are tried next to this many nearest villages
SEED = 12345


def savings(n, d, dem, Q):
    routes = {v: [v] for v in range(1, n + 1)}
    rid = list(range(n + 1))
    load = list(dem)
    sav = sorted(((d[0][i] + d[0][j] - d[i][j], i, j)
                  for i in range(1, n + 1) for j in range(i + 1, n + 1)), reverse=True)
    for s, i, j in sav:
        if s <= 0:
            break
        ri, rj = rid[i], rid[j]
        if ri == rj or load[ri] + load[rj] > Q:
            continue
        A, B = routes[ri], routes[rj]
        if A[-1] == i and B[0] == j:
            new = A + B
        elif A[0] == i and B[-1] == j:
            new = B + A
        elif A[0] == i and B[0] == j:
            new = A[::-1] + B
        elif A[-1] == i and B[-1] == j:
            new = A + B[::-1]
        else:
            continue
        routes[ri] = new
        del routes[rj]
        load[ri] += load[rj]
        for x in B:
            rid[x] = ri
    return list(routes.values())


class MineSolver(Solver):
    def solve(self, instance, submit_candidate, start_routes=None, t_start=None, t_end=None, quiet=False):
        t0 = time.perf_counter() if t_start is None else t_start
        n, Q, K = instance.size, instance.capacity, instance.fleet
        dem = instance.demand
        d = distance_matrix(instance)
        d0 = d[0]
        rnd = random.Random(SEED)
        rand = rnd.random
        randint = rnd.randint

        routes = start_routes
        if routes is None:
            routes = savings(n, d, dem, Q)
            if len(routes) > K:
                routes = StarterSolver().solve(instance, submit_candidate)["routes"]
        R = [list(r) for r in routes if r]
        rt = [0] * (n + 1)
        loads = []
        for k, r in enumerate(R):
            loads.append(sum(dem[v] for v in r))
            for v in r:
                rt[v] = k

        def rlen(r):
            if not r:
                return 0
            c = d0[r[0]] + d0[r[-1]]
            for a, b in zip(r, r[1:]):
                c += d[a][b]
            return c

        cur = sum(rlen(r) for r in R)
        best = cur
        best_routes = [r[:] for r in R]
        if start_routes is None:
            submit_candidate({"routes": [r[:] for r in best_routes]})
        submitted_best = best
        last_submit = time.perf_counter()

        full = [None] + [[u for u in sorted(range(1, n + 1), key=d[v].__getitem__) if u != v] for v in range(1, n + 1)]
        nbr = [None] + [full[v][:NB] for v in range(1, n + 1)]
        avg_leg = max(1.0, cur / (n + len(R)))
        T0, T1 = T0_FACTOR * avg_leg, T1_FACTOR * avg_leg
        end_t = (t0 + STOP_AT_S) if t_end is None else t_end
        cbar = min(CBAR, max(3, n // 5))
        far_key = lambda c: -d0[c]
        it = 0
        T = T0
        sa_start = time.perf_counter()
        exp, log = math.exp, math.log

        while True:
            it += 1
            if it & 63 == 0:
                now = time.perf_counter()
                if now >= end_t:
                    break
                T = T0 * (T1 / T0) ** ((now - sa_start) / (end_t - sa_start))
                if best < submitted_best and now - last_submit >= SUBMIT_EVERY_S and not quiet:
                    submit_candidate({"routes": [r[:] for r in best_routes]})
                    submitted_best, last_submit = best, now

            # ---------------- snapshot (copy-on-write) ----------------
            saved = {}
            saved_rt = []
            nR = len(R)
            delta = 0

            # ---------------- ruin ----------------
            avg_card = n / max(1, sum(1 for r in R if r))
            lsmax = min(LMAX, avg_card)
            ksmax = 4.0 * cbar / (1.0 + lsmax) - 1.0
            ks = int(rand() * ksmax + 1.0)
            seed = randint(1, n)
            ruined = set()
            removed = []
            for c in [seed] + full[seed]:
                if len(ruined) >= ks:
                    break
                A = rt[c]
                if A in ruined or rt[c] < 0:
                    continue
                RA = R[A]
                ltmax = min(len(RA), lsmax)
                lt = int(rand() * ltmax + 1.0)
                pos = RA.index(c)
                lo = max(0, pos - lt + 1)
                hi = min(pos, len(RA) - lt)
                st = randint(lo, hi) if hi > lo else lo
                if A not in saved:
                    saved[A] = (RA[:], loads[A])
                seg = RA[st:st + lt]
                # delta of removal
                p = RA[st - 1] if st else 0
                s = RA[st + lt] if st + lt < len(RA) else 0
                delta -= (d[p][seg[0]] + d[seg[-1]][s] - d[p][s])
                for a, b in zip(seg, seg[1:]):
                    delta -= d[a][b]
                del RA[st:st + lt]
                for x in seg:
                    saved_rt.append((x, A))
                    rt[x] = -1
                    loads[A] -= dem[x]
                removed.extend(seg)
                ruined.add(A)

            # ---------------- recreate ----------------
            r = rand() * 11
            if r < 4:
                rnd.shuffle(removed)
            elif r < 8:
                removed.sort(key=lambda c: -dem[c])
            elif r < 10:
                removed.sort(key=far_key)
            else:
                removed.sort(key=d0.__getitem__)

            used = sum(1 for x in R if x)
            ok = True
            for c in removed:
                dc = dem[c]
                bestc = None
                bestR = -1
                bestpos = 0
                for u in nbr[c]:
                    B = rt[u]
                    if B < 0 or loads[B] + dc > Q:
                        continue
                    RB = R[B]
                    j = RB.index(u)
                    # after u
                    if rand() >= BLINK:
                        y = RB[j + 1] if j + 1 < len(RB) else 0
                        cc = d[u][c] + d[c][y] - d[u][y]
                        if bestc is None or cc < bestc:
                            bestc, bestR, bestpos = cc, B, j + 1
                    # before u
                    if rand() >= BLINK:
                        x = RB[j - 1] if j else 0
                        cc = d[x][c] + d[c][u] - d[x][u]
                        if bestc is None or cc < bestc:
                            bestc, bestR, bestpos = cc, B, j
                if used < K:
                    cc = 2 * d0[c]
                    if bestc is None or cc < bestc:
                        bestc, bestR, bestpos = cc, -2, 0
                if bestc is None:
                    # full scan fallback
                    for B in range(len(R)):
                        if loads[B] + dc > Q:
                            continue
                        RB = R[B]
                        if not RB:
                            continue
                        for j in range(len(RB) + 1):
                            x = RB[j - 1] if j else 0
                            y = RB[j] if j < len(RB) else 0
                            cc = d[x][c] + d[c][y] - d[x][y]
                            if bestc is None or cc < bestc:
                                bestc, bestR, bestpos = cc, B, j
                if bestc is None:
                    ok = False
                    break
                if bestR == -2:
                    # open a new route (reuse an empty slot if any)
                    e = -1
                    for k in range(len(R)):
                        if not R[k]:
                            e = k
                            break
                    if e < 0:
                        R.append([])
                        loads.append(0)
                        e = len(R) - 1
                    if e not in saved:
                        saved[e] = ([], 0)
                    R[e] = [c]
                    loads[e] = dc
                    rt[c] = e
                    used += 1
                else:
                    if bestR not in saved:
                        saved[bestR] = (R[bestR][:], loads[bestR])
                    was_empty = not R[bestR]
                    R[bestR].insert(bestpos, c)
                    loads[bestR] += dc
                    rt[c] = bestR
                    if was_empty:
                        used += 1
                delta += bestc

            if ok:
                new = cur + delta
                if new < cur or rand() < exp(-(delta) / T):
                    accept = True
                else:
                    accept = False
            else:
                accept = False

            if accept:
                cur += delta
                if cur < best:
                    best = cur
                    best_routes = [r[:] for r in R if r]
            else:
                for k, (lst, ld) in saved.items():
                    if k < len(R):
                        R[k] = lst
                        loads[k] = ld
                for x, A in saved_rt:
                    rt[x] = A
                # routes appended during recreate that are now stale keep empty content
                for k in range(len(R)):
                    pass
                # fix rt for customers moved into other routes: restored via saved lists
                for k, (lst, ld) in saved.items():
                    for x in lst:
                        rt[x] = k
        return {"routes": best_routes}