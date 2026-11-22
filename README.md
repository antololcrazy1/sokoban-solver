# Sokoban Solver

A push-optimal Sokoban solver, built for make.fun's
`sokoban-solver-01-run-20261121` bounty ("Build the best Sokoban solver, 21
November 2026").

## How it works

The score is per push, not per move, and a level is worth its full 1,000
points only in P\* pushes — one push over already halves it. So this program
searches over pushes, and searches for the fewest of them.

Generated levels are a 1-wide hallway with 1–4 rooms off it (3–4 boxes each).
A box pushed through a door into the hallway can never come back, so every
box stays in its room and the level optimum is the sum of the rooms' optima.
Each room is therefore solved on its own with A\* over pushes:

- a state is the boxes' squares plus the player's reachable region (named by
  its lowest square), so the player never walks explicitly;
- the heuristic is the cheapest assignment of boxes to goals by single-box
  push distance (a lone box, other boxes ignored), which never overestimates:
  one push moves one box one step;
- dead squares, frozen 2×2 blocks and frozen boxes prune dead states without
  ever pruning a solvable one.

With a consistent heuristic the first solved state A\* pops is push-optimal,
so every room — and therefore the level — is solved in P\*. If A\* ever
exhausts its work budget on a room, a plain push-BFS runs as a fallback.
Work budgets are counts of expanded states, never the clock, so two runs
write the same bytes.

## Run

```sh
node main.ts   # reads /input/<id>/level.txt, writes /output/<id>/solution.txt
```

Node 22 runs `main.ts` as it is (erasable TypeScript only). One run answers
the whole set of 5 instances well inside the 60-second wall budget.

## Results (practice set, kit scorer)

| instance | P\* | pushes | points |
|----------|----|--------|--------|
| i001 | 13 | 13 | 1,000 |
| i002 | 16 | 16 | 1,000 |
| i003 | 29 | 29 | 1,000 |
| i004 | 47 | 47 | 1,000 |
| i005 | 60 | 60 | 1,000 |

**Score: 1,000,000 / 1,000,000** (house baseline plain-astar 1.0.0 scores
200,000 on the same set). Also 1,000,000 on 6 freshly drawn sets from the
kit generator (different seeds), each solved in under 1.1 s.

## License

MIT — see `LICENSE`.
