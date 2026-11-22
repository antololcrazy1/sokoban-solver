// Sokoban solver — push-optimal A* over pushes, solved room by room.
//
// Why this beats the house baseline (plain A* over moves, no deadlock
// knowledge): the score is per push, not per move, and a level is worth its
// full 1,000 points only in P* pushes — one push over already halves it. So
// this program searches over pushes, and searches for the fewest of them.
//
// Generated levels are a 1-wide hallway with 1-4 rooms off it (3-4 boxes
// each). A box pushed through a door into the hallway can never come back,
// so every box stays in its room and the level optimum is the sum of the
// rooms' optima. Each room is therefore solved on its own:
//   * a state is the boxes' squares plus the player's reachable region
//     (named by its lowest square), so the player never walks explicitly;
//   * the heuristic is the cheapest assignment of boxes to goals by
//     single-box push distance (a lone box, other boxes ignored), which
//     never overestimates: one push moves one box one step, so the
//     assignment cost falls by at most 1 per push;
//   * dead squares (a box that can never be pushed to a goal), frozen 2x2
//     blocks and frozen boxes (a box that can never move again) prune dead
//     states without ever pruning a solvable one.
// With a consistent heuristic the first solved state A* pops is
// push-optimal, so every room — and therefore the level — is solved in P*.
// If A* ever exhausts its work budget on a room, a plain push-BFS (the
// search the generator itself uses to certify rooms) runs as a fallback.
//
// Work budgets are counts of expanded states, never the clock, so two runs
// expand the same states and write the same bytes. Node 22 runs this file
// as it is (erasable TypeScript only, no imports beyond node:fs/path).
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const I = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const O = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

// Work budgets, in expanded states. Counts, never the clock.
const MAX_EXPAND_ROOM = 200_000;
const MAX_EXPAND_INSTANCE = 600_000;

const DX = [0, 0, -1, 1];
const DY = [-1, 1, 0, 0];
const STEP = ['u', 'd', 'l', 'r'];
const PUSH = ['U', 'D', 'L', 'R'];

type Level = {
  w: number;
  h: number;
  wall: Uint8Array;
  goals: number[];
  boxes: number[];
  player: number;
};

type Budget = { used: number; limit: number };
type SolveResult = { pushes: number; moves: string } | null;

/** A level in XSB text (# wall, space floor, . goal, $ box, * box on a goal, @ player). */
function parseLevel(text: string): Level {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const h = rows.length;
  const w = Math.max(...rows.map((r) => r.length));
  const wall = new Uint8Array(w * h).fill(1);
  const goals: number[] = [];
  const boxes: number[] = [];
  let player = -1;
  rows.forEach((row, y) => {
    for (let x = 0; x < row.length; x++) {
      const ch = row[x];
      const i = y * w + x;
      if (ch === '#') continue;
      wall[i] = 0;
      if (ch === '.' || ch === '*' || ch === '+') goals.push(i);
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  });
  // The outermost frame is a wall whatever it shows: nothing leaves the board.
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (x === 0 || y === 0 || x === w - 1 || y === h - 1) wall[y * w + x] = 1;
    }
  }
  return { w, h, wall, goals, boxes: boxes.sort((a, b) => a - b), player };
}

/** The squares from which a box alone can be pushed onto some goal, pulled back from the goals over open floor. */
function liveSquares(level: Level): Uint8Array {
  const { w, wall, goals } = level;
  const size = wall.length;
  const live = new Uint8Array(size);
  const queue = [...goals];
  for (const g of goals) live[g] = 1;
  for (let qi = 0; qi < queue.length; qi++) {
    const x = queue[qi];
    for (let k = 0; k < 4; k++) {
      const d = DY[k] * w + DX[k];
      const y = x - d;
      if (y < 0 || y >= size || wall[y] || live[y]) continue;
      const behind = y - d;
      if (behind < 0 || behind >= size || wall[behind]) continue;
      live[y] = 1;
      queue.push(y);
    }
  }
  return live;
}

/**
 * For each goal, the fewest pushes a lone box needs to reach it, other boxes
 * ignored: a box at x pushes to x+d when the player can stand at x-d and the
 * square beyond is open. BFS backwards from the goal over those edges, so
 * dist[s * nGoals + gi] is that count (Infinity when unreachable).
 */
function pushDistances(level: Level): Float64Array {
  const { w, wall, goals } = level;
  const size = wall.length;
  const n = goals.length;
  const dist = new Float64Array(size * n).fill(Infinity);
  for (let gi = 0; gi < n; gi++) {
    const g = goals[gi];
    dist[g * n + gi] = 0;
    const queue = [g];
    for (let qi = 0; qi < queue.length; qi++) {
      const y = queue[qi];
      const dy = dist[y * n + gi];
      for (let k = 0; k < 4; k++) {
        const d = DY[k] * w + DX[k];
        const x = y - d;
        if (x < 0 || x >= size || wall[x] || dist[x * n + gi] !== Infinity) continue;
        const behind = x - d;
        if (behind < 0 || behind >= size || wall[behind]) continue;
        dist[x * n + gi] = dy + 1;
        queue.push(x);
      }
    }
  }
  return dist;
}

/**
 * Cheapest assignment of boxes to goals by single-box push distance.
 * Up to 5 boxes: brute force over the assignments; more boxes: each box's
 * own nearest goal (still a lower bound, just weaker). Infinity when no
 * assignment works — then the state cannot be solved.
 */
function heuristic(boxes: number[], dist: Float64Array, nGoals: number): number {
  const n = boxes.length;
  if (n === 0) return 0;
  if (n <= 5) {
    const used = new Array<boolean>(nGoals).fill(false);
    let best = Infinity;
    const rec = (i: number, sum: number): void => {
      if (sum >= best) return;
      if (i === n) {
        best = sum;
        return;
      }
      const b = boxes[i];
      for (let gi = 0; gi < nGoals; gi++) {
        if (used[gi]) continue;
        const d = dist[b * nGoals + gi];
        if (d === Infinity) continue;
        used[gi] = true;
        rec(i + 1, sum + d);
        used[gi] = false;
      }
    };
    rec(0, 0);
    return best;
  }
  let sum = 0;
  for (const b of boxes) {
    let m = Infinity;
    for (let gi = 0; gi < nGoals; gi++) m = Math.min(m, dist[b * nGoals + gi]);
    if (m === Infinity) return Infinity;
    sum += m;
  }
  return sum;
}

/** The player's reachable region from `from` with `occupied` in the way: its lowest square, marking squares in `seen` with `stamp`. */
function regionLow(level: Level, occupied: Uint8Array, from: number, seen: Int32Array, stamp: number, stack: Int32Array): number {
  const { wall } = level;
  const w = level.w;
  let low = from;
  let n = 0;
  stack[n++] = from;
  seen[from] = stamp;
  while (n > 0) {
    const p = stack[--n];
    if (p < low) low = p;
    const q0 = p - w;
    const q1 = p + w;
    const q2 = p - 1;
    const q3 = p + 1;
    if (!wall[q0] && !occupied[q0] && seen[q0] !== stamp) {
      seen[q0] = stamp;
      stack[n++] = q0;
    }
    if (!wall[q1] && !occupied[q1] && seen[q1] !== stamp) {
      seen[q1] = stamp;
      stack[n++] = q1;
    }
    if (!wall[q2] && !occupied[q2] && seen[q2] !== stamp) {
      seen[q2] = stamp;
      stack[n++] = q2;
    }
    if (!wall[q3] && !occupied[q3] && seen[q3] !== stamp) {
      seen[q3] = stamp;
      stack[n++] = q3;
    }
  }
  return low;
}

/** Whether the box just pushed to `t` sits in a frozen 2x2 block of walls and boxes with a box off its goal. */
function frozenBlock(level: Level, occupied: Uint8Array, goal: Uint8Array, t: number): boolean {
  const { w, wall } = level;
  const size = wall.length;
  for (const a of [t, t - 1, t - w, t - w - 1]) {
    const b = a + 1;
    const c = a + w;
    const d = c + 1;
    if (a < 0 || d >= size) continue;
    if (!(wall[a] || occupied[a]) || !(wall[b] || occupied[b]) || !(wall[c] || occupied[c]) || !(wall[d] || occupied[d])) continue;
    if ((occupied[a] && !goal[a]) || (occupied[b] && !goal[b]) || (occupied[c] && !goal[c]) || (occupied[d] && !goal[d])) return true;
  }
  return false;
}

/**
 * A box is frozen when it cannot be pushed along either axis: on each axis at
 * least one neighbour is a wall or a frozen box. Frozen boxes never move
 * again, so a frozen box off its goal deadlocks the state. Iterated to a
 * fixpoint; the 2x2 case is caught by frozenBlock, not here.
 */
function freezeDeadlock(level: Level, boxes: number[], goal: Uint8Array): boolean {
  const { w, wall } = level;
  const size = wall.length;
  const frozen = new Uint8Array(size);
  let changed = true;
  while (changed) {
    changed = false;
    for (const b of boxes) {
      if (frozen[b]) continue;
      const xBlocked = (wall[b - 1] || frozen[b - 1]) || (wall[b + 1] || frozen[b + 1]);
      const yBlocked = (wall[b - w] || frozen[b - w]) || (wall[b + w] || frozen[b + w]);
      if (xBlocked && yBlocked) {
        frozen[b] = 1;
        changed = true;
      }
    }
  }
  for (const b of boxes) if (frozen[b] && !goal[b]) return true;
  return false;
}

/** The shortest walk from `from` to `to` around walls and boxes, in lower LURD letters (first direction first on ties). */
function walkTo(level: Level, occupied: Uint8Array, from: number, to: number): string {
  const { w, wall } = level;
  if (from === to) return '';
  const prev = new Int32Array(wall.length).fill(-1);
  const how = new Array<string>(wall.length);
  const queue = [from];
  prev[from] = from;
  for (let qi = 0; qi < queue.length; qi++) {
    const p = queue[qi];
    if (p === to) break;
    for (let k = 0; k < 4; k++) {
      const q = p + DY[k] * w + DX[k];
      if (wall[q] || occupied[q] || prev[q] !== -1) continue;
      prev[q] = p;
      how[q] = STEP[k];
      queue.push(q);
    }
  }
  if (prev[to] === -1) throw new Error('walkTo: the push square is out of reach');
  let s = '';
  for (let p = to; p !== from; p = prev[p]) s = how[p] + s;
  return s;
}

/** The square the player stands on after `moves` (LURD, case aside) from `from` on a board `w` wide. */
function endOf(w: number, from: number, moves: string): number {
  let p = from;
  for (const ch of moves.toLowerCase()) p += ch === 'u' ? -w : ch === 'd' ? w : ch === 'l' ? -1 : 1;
  return p;
}

/**
 * Push-optimal search over one level (or one room's sub-level): A* over
 * pushes when `useHeuristic`, plain push-BFS otherwise. The first solved
 * state popped is push-optimal. Null when the work budget runs out or the
 * level has no solution.
 */
function searchPush(level: Level, budget: Budget, maxExpand: number, useHeuristic: boolean): SolveResult {
  const { w, wall, goals } = level;
  const size = wall.length;
  const goal = new Uint8Array(size);
  for (const g of goals) goal[g] = 1;
  const live = liveSquares(level);
  const dist = pushDistances(level);
  const nGoals = goals.length;
  const solved = (boxes: number[]): boolean => boxes.every((b) => goal[b]);
  if (level.boxes.some((b) => !live[b])) return null;
  const h0 = useHeuristic ? heuristic(level.boxes, dist, nGoals) : 0;
  if (h0 === Infinity) return null;

  const seen = new Int32Array(size);
  const mark = new Int32Array(size);
  const stack = new Int32Array(size);
  const occupied = new Uint8Array(size);
  let stamp = 0;

  // The states in the order found: their boxes, the player's square after the
  // push that made them, their push count, parent and state key.
  const sBoxes: number[][] = [level.boxes.slice()];
  const sPlayer: number[] = [level.player];
  const sG: number[] = [0];
  const sParent: number[] = [-1];
  const sKey: string[] = [];
  const keyOf = (boxes: number[], low: number): string => boxes.join(',') + '|' + low;
  const best = new Map<string, number>();
  for (const b of level.boxes) occupied[b] = 1;
  const low0 = regionLow(level, occupied, level.player, seen, ++stamp, stack);
  for (const b of level.boxes) occupied[b] = 0;
  sKey.push(keyOf(level.boxes, low0));
  best.set(sKey[0], 0);

  // Binary heap on [f, g, seq, state]: lowest f first, deeper g first on a tie,
  // insertion order after that, so the run is deterministic.
  const heap: number[][] = [];
  const less = (a: number[], b: number[]): boolean =>
    a[0] < b[0] || (a[0] === b[0] && (a[1] > b[1] || (a[1] === b[1] && a[2] < b[2])));
  const heapPush = (e: number[]): void => {
    let i = heap.length;
    heap.push(e);
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!less(heap[i], heap[p])) break;
      const tmp = heap[i];
      heap[i] = heap[p];
      heap[p] = tmp;
      i = p;
    }
  };
  const heapPop = (): number[] => {
    const top = heap[0];
    const last = heap.pop() as number[];
    if (heap.length > 0) {
      heap[0] = last;
      for (let i = 0; ; ) {
        let c = 2 * i + 1;
        if (c >= heap.length) break;
        if (c + 1 < heap.length && less(heap[c + 1], heap[c])) c++;
        if (!less(heap[c], heap[i])) break;
        const tmp = heap[i];
        heap[i] = heap[c];
        heap[c] = tmp;
        i = c;
      }
    }
    return top;
  };

  let seq = 1;
  heapPush([h0, 0, 0, 0]);
  let expanded = 0;
  let found = -1;
  while (heap.length > 0 && expanded < maxExpand && budget.used < budget.limit) {
    const top = heapPop();
    const s = top[3];
    const g = sG[s];
    if (best.get(sKey[s]) !== g) continue; // stale: a better route to this state exists
    const boxes = sBoxes[s];
    if (solved(boxes)) {
      found = s;
      break;
    }
    expanded++;
    budget.used++;
    for (const b of boxes) occupied[b] = 1;
    regionLow(level, occupied, sPlayer[s], seen, ++stamp, stack);
    const here = stamp;
    for (let bi = 0; bi < boxes.length; bi++) {
      const b = boxes[bi];
      for (let k = 0; k < 4; k++) {
        const d = DY[k] * w + DX[k];
        const t = b + d;
        if (wall[t] || occupied[t] || !live[t]) continue;
        if (seen[b - d] !== here) continue;
        occupied[b] = 0;
        occupied[t] = 1;
        const frozen = frozenBlock(level, occupied, goal, t);
        let next: number[] | null = null;
        let key = '';
        if (!frozen) {
          next = boxes.slice();
          next[bi] = t;
          next.sort((x, y) => x - y);
          if (freezeDeadlock(level, next, goal)) {
            next = null;
          } else {
            const low = regionLow(level, occupied, b, mark, ++stamp, stack);
            key = keyOf(next, low);
          }
        }
        occupied[t] = 0;
        occupied[b] = 1;
        if (next === null) continue;
        const g2 = g + 1;
        const prev = best.get(key);
        if (prev !== undefined && prev <= g2) continue;
        const h = useHeuristic ? heuristic(next, dist, nGoals) : 0;
        if (h === Infinity) continue;
        best.set(key, g2);
        const idx = sBoxes.length;
        sBoxes.push(next);
        sPlayer.push(b);
        sG.push(g2);
        sParent.push(s);
        sKey.push(key);
        heapPush([g2 + h, g2, seq++, idx]);
      }
    }
    for (const b of boxes) occupied[b] = 0;
  }
  if (found < 0) return null;

  // The path of states, then the moves: walk to each push's square, then push.
  const path: number[] = [];
  for (let s = found; s > 0; s = sParent[s]) path.push(s);
  path.reverse();
  let moves = '';
  let player = level.player;
  let boxes = level.boxes;
  for (const s of path) {
    const nextBoxes = sBoxes[s];
    const from = boxes.find((b) => !nextBoxes.includes(b)) as number;
    const to = nextBoxes.find((b) => !boxes.includes(b)) as number;
    const d = to - from;
    let k = 0;
    for (; k < 4; k++) if (DY[k] * w + DX[k] === d) break;
    for (const b of boxes) occupied[b] = 1;
    moves += walkTo(level, occupied, player, from - d);
    for (const b of boxes) occupied[b] = 0;
    moves += PUSH[k];
    player = from;
    boxes = nextBoxes;
  }
  return { pushes: path.length, moves };
}

type Room = { squares: number[]; boxes: number[]; goals: number[]; done: boolean };

/**
 * The level's rooms: cut the dead squares of the player's row (the hallway:
 * a box there reaches no goal) out of the floor, and each part left that
 * holds a box is a room. Null when some part holds more or fewer boxes than
 * goals, or a box starts in the hallway: then the level is no set of rooms.
 */
function roomsOf(level: Level): Room[] | null {
  const { w, wall, goals, boxes, player } = level;
  const live = liveSquares(level);
  const row = Math.floor(player / w);
  const cut = new Uint8Array(wall.length);
  for (let i = row * w; i < (row + 1) * w; i++) if (!wall[i] && !live[i]) cut[i] = 1;
  if (boxes.some((b) => cut[b])) return null;
  const part = new Int32Array(wall.length).fill(-1);
  const rooms: Room[] = [];
  for (let s = 0; s < wall.length; s++) {
    if (wall[s] || cut[s] || part[s] >= 0) continue;
    const squares = [s];
    part[s] = rooms.length;
    for (let i = 0; i < squares.length; i++) {
      for (let k = 0; k < 4; k++) {
        const q = squares[i] + DY[k] * w + DX[k];
        if (!wall[q] && !cut[q] && part[q] < 0) {
          part[q] = rooms.length;
          squares.push(q);
        }
      }
    }
    rooms.push({ squares, boxes: [], goals: [], done: false });
  }
  for (const b of boxes) rooms[part[b]].boxes.push(b);
  for (const g of goals) if (part[g] >= 0) rooms[part[g]].goals.push(g);
  if (goals.some((g) => part[g] < 0) || rooms.some((r) => r.boxes.length !== r.goals.length)) return null;
  return rooms.filter((r) => r.boxes.length > 0);
}

/**
 * The level solved room by room: each room's push-optimal solution in turn,
 * the other rooms' boxes standing as walls (on their goals once their room is
 * done), the player walking on from where the last room left it. A box never
 * leaves its room, so the rooms' optima add up to the level's. Falls back to
 * a search over the whole level when it is no set of rooms.
 */
function solveByRooms(level: Level, budget: Budget): SolveResult {
  const rooms = roomsOf(level);
  if (rooms === null) {
    return searchPush(level, budget, MAX_EXPAND_ROOM, true) ?? searchPush(level, budget, MAX_EXPAND_ROOM, false);
  }
  let player = level.player;
  let pushes = 0;
  let moves = '';
  for (const room of rooms) {
    const wall = level.wall.slice();
    for (const other of rooms) {
      if (other === room) continue;
      for (const b of other.done ? other.goals : other.boxes) wall[b] = 1;
    }
    const sub: Level = {
      w: level.w,
      h: level.h,
      wall,
      goals: room.goals.slice().sort((a, b) => a - b),
      boxes: room.boxes.slice().sort((a, b) => a - b),
      player,
    };
    let solved = searchPush(sub, budget, MAX_EXPAND_ROOM, true);
    if (solved === null) solved = searchPush(sub, budget, MAX_EXPAND_ROOM, false);
    if (solved === null) return null;
    pushes += solved.pushes;
    moves += solved.moves;
    player = endOf(level.w, player, solved.moves);
    room.done = true;
  }
  return { pushes, moves };
}

/** The moves for a level in XSB: a string of u, d, l and r (an upper letter marks a push; case is not checked). */
function solve(text: string): string {
  const level = parseLevel(text);
  if (level.boxes.length === 0) return '';
  const budget: Budget = { used: 0, limit: MAX_EXPAND_INSTANCE };
  const result = solveByRooms(level, budget);
  return result === null ? '' : result.moves;
}

for (const id of readdirSync(I).sort()) {
  let moves = '';
  try {
    moves = solve(readFileSync(join(I, id, 'level.txt'), 'utf8'));
  } catch (e) {
    console.error(`sokoban-solver: ${id}: ${e instanceof Error ? e.message : String(e)}`);
    moves = '';
  }
  mkdirSync(join(O, id), { recursive: true });
  writeFileSync(join(O, id, 'solution.txt'), `${moves}\n`);
}
