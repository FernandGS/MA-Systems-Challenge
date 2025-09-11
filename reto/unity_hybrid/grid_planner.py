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


def _draw_disk(mask: List[List[int]], cx: int, cy: int, r: int):
    H, W = len(mask), len(mask[0])
    r2 = r*r
    for dy in range(-r, r+1):
        yy = cy + dy
        if yy < 0 or yy >= H:
            continue
        # horizontal span for this dy
        max_dx = int((r2 - dy*dy) ** 0.5)
        x0, x1 = cx - max_dx, cx + max_dx
        if x1 < 0 or x0 >= W:
            continue
        x0 = 0 if x0 < 0 else x0
        x1 = W-1 if x1 >= W else x1
        for xx in range(x0, x1+1):
            mask[yy][xx] = 1


def city_to_grid_mask(city, N: int, extra_margin_m: float = 0.5) -> List[List[int]]:
    """Rasterize the City's road centerlines to a passable grid of size NxN.
    Expands roads by ROAD_HALF_WIDTH + extra_margin_m in world units.
    """
    Wm, Hm = float(city.w), float(city.h)
    sx = N / Wm if Wm > 0 else 1.0
    sy = N / Hm if Hm > 0 else 1.0
    mask = [[0]*N for _ in range(N)]
    road_half = float(getattr(city, 'cfg', {}).get('ROAD_HALF_WIDTH', 3.5))
    radius_m = max(0.5, road_half + float(extra_margin_m))
    # pixel radius conservative: use max scale to over-cover
    rpix = max(1, int(math.ceil(radius_m * max(sx, sy))))
    for r in city.roads:
        (x1,y1),(x2,y2) = r.polyline
        ix1, iy1 = int(round(x1 * sx)), int(round(y1 * sy))
        ix2, iy2 = int(round(x2 * sx)), int(round(y2 * sy))
        dx, dy = ix2 - ix1, iy2 - iy1
        steps = max(abs(dx), abs(dy), 1)
        for k in range(steps+1):
            t = 0.0 if steps == 0 else k / steps
            ix = int(round(ix1 + t * dx))
            iy = int(round(iy1 + t * dy))
            _draw_disk(mask, ix, iy, rpix)
    return mask


def save_mask_png(mask: List[List[int]], path: str):
    if Image is None:
        return
    H, W = len(mask), len(mask[0])
    img = Image.new('L', (W, H))
    px = img.load()
    for y in range(H):
        for x in range(W):
            px[x, y] = 255 if mask[y][x] == 1 else 0
    img.save(path)
