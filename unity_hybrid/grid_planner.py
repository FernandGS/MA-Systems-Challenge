from typing import List, Tuple, Optional
import math

PointI = Tuple[int, int]

try:
    from PIL import Image
except Exception:
    Image = None


def load_mask_png(path: str, N: int, threshold: float = 0.5, invert_y: bool = False) -> List[List[int]]:
    if Image is None:
        raise RuntimeError("Pillow not installed; cannot load PNG mask.")
    img = Image.open(path).convert("L").resize((N, N))
    px = img.load()
    grid = [[0]*N for _ in range(N)]
    for y in range(N):
        ry = (N-1-y) if invert_y else y
        for x in range(N):
            val = px[x, ry] / 255.0
            grid[y][x] = 1 if val >= threshold else 0
    return grid


def neighbors4(x: int, y: int, W: int, H: int):
    if x>0: yield (x-1,y)
    if x<W-1: yield (x+1,y)
    if y>0: yield (x,y-1)
    if y<H-1: yield (x,y+1)


def astar(start: PointI, goal: PointI, passable: List[List[int]]) -> List[PointI]:
    W, H = len(passable[0]), len(passable)
    def h(a: PointI, b: PointI): return abs(a[0]-b[0]) + abs(a[1]-b[1])
    openq = [(h(start, goal), 0, start, None)]
    came, g = {}, {start:0}
    import heapq
    while openq:
        f, gc, node, parent = heapq.heappop(openq)
        if node in came: continue
        came[node] = parent
        if node == goal: break
        x,y = node
        for nx,ny in neighbors4(x,y,W,H):
            if passable[ny][nx] != 1: continue
            ng = gc + 1
            if (nx,ny) not in g or ng < g[(nx,ny)]:
                g[(nx,ny)] = ng
                heapq.heappush(openq, (ng + h((nx,ny), goal), ng, (nx,ny), node))
    if goal not in came: return []
    path = []
    u = goal
    while u is not None:
        path.append(u); u = came[u]
    path.reverse(); return path


def manhattan_path(start: PointI, goal: PointI) -> List[PointI]:
    (x,y),(gx,gy) = start, goal
    path = [(x,y)]
    while x != gx:
        x += 1 if gx>x else -1
        path.append((x,y))
    while y != gy:
        y += 1 if gy>y else -1
        path.append((x,y))
    return path


def dilate_mask(mask: List[List[int]]) -> List[List[int]]:
    H, W = len(mask), len(mask[0])
    out = [[0]*W for _ in range(H)]
    for y in range(H):
        for x in range(W):
            if mask[y][x] == 1:
                out[y][x] = 1
                for nx,ny in neighbors4(x,y,W,H):
                    out[ny][nx] = 1
    return out
