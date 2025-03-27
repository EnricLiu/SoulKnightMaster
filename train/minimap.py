from math import pi as PI
from pathlib import Path
from heapq import heappush, heappop

import numpy as np
from PIL import Image
from PIL.Image import Image as _Image

import torch

from utils import Position, dhash, dhash_batch
    
class SoulKnightMinimap:
    def __init__(self, config):
        self.config = config
        self.ckpts_root = config["root_path"]
        room_config = config["room"]
        road_config = config["road"]
        
        def get_ckpt_path(p: str) -> Path:
            ret = Path(f"{self.ckpts_root}/{p}")
            if not ret.is_file():
                raise ValueError(f"{p} is not exists")
            return ret

        room_types  = [(None, None, None)]
        thresh_vecs = []
        hash_vecs   = []
        for room_type, status_ckpt_dict in config["ckpts"].items():
            for status, details in status_ckpt_dict.items():
                room_types.append((room_type, status, details["cost"]))
                thresh_vecs.append(details["threshold"])
                hash_vecs.append(dhash(np.array(Image.open(get_ckpt_path(details["image"]))), None))
        
        self.ckpts = {
            "room_types" :    room_types,
            "thresh_vecs" :   np.array(thresh_vecs),                        # [ckpt_len]
            "hash_vecs" :     np.stack(hash_vecs, axis=0)[np.newaxis,:],    # [1, ckpt_len, 1102]
        }
        
        self.room = {
            "mask":     np.array(Image.open(get_ckpt_path(room_config["mask"]))) == 0,
            "size":     room_config["size"],
            "interval": room_config["interval"],
            "tl_pos":   Position(room_config["first_pos"]["x"], room_config["first_pos"]["y"]),
        }
        self.road = {
            "width":    road_config["width"],
            "length":   road_config["length"],
            "offset":   road_config["offset"],
            "color":    np.array(road_config["color"]),
        }
        
        self.region = {
            "tl": Position(config["region"]["tl"]["x"], config["region"]["tl"]["y"]),
            "br": Position(config["region"]["br"]["x"], config["region"]["br"]["y"]),
        }
        
        # print(self.ckpts)
        self.arrows = None # lazy loading
        self.icons  = None # lazy loading
        self.graph: SoulKnightGraph = None
        self.reset()
    
    def reset(self):
        self.graph = SoulKnightGraph(0, self.ckpts["room_types"])
    
    def update(self, mini_map: np.ndarray):
        rooms, roads, _ = self.parse(mini_map)
        res, info = self.graph.update(rooms, roads)
        if not res and info == "Center Not Detected":
            self.reset()
        return res
    
    def parse(self, mini_map: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """parse the given minimap to rooms and roads

        Args:
            mini_map (np.ndarray): (200, 200, 3)

        Returns:
            tuple[np.ndarray, np.ndarray]: rooms, roads, (5, 5)
        """
        rooms, mask = self._extract_room(mini_map)
        roads = self._extract_road(mini_map, mask)
        
        if rooms is None: 
            rooms = np.zeros((5,5), dtype=np.uint8)
        if roads is None: 
            roads = np.zeros((5,5), dtype=np.uint8)

        current_pos = np.zeros((5,5), dtype=np.uint8)
        current_pos[2,2] = 1
            
        return rooms, roads, current_pos
    
    def navigate(self) -> float:
        for t in ["portal", "boss"]:
            for room in self.ckpts["room_types"]:
                room_type, room_status, _ = room
                if room_type == t:
                    path = self.graph.find_path(room_type, room_status)
                    if path is not None: return self._path_to_direction(path)
        
        for t in ["enemy", "jail"]:
            path = self.graph.find_path(t, "veiled")
            if path is None: continue
            return self._path_to_direction(path)
        return None
    
    def render(self):
        img = self._render()
        return Image.fromarray(img) if img is not None else None
    
    def _render(self):
        nodes = self.graph.node_graph
        roads = self.graph.access_graph
        if nodes is None or roads is None: return None
        
        if self.arrows is None:
            self.arrows = [np.array(Image.open(Path(f"{self.ckpts_root}/arrow_{dir}.png")))[:,:,0] == 0 for dir in ["up", "down", "left", "right"]]
            self.icons = [np.zeros((24,24,3),dtype=np.uint8)]
            for room_type, status_ckpt_dict in config["ckpts"].items():
                for status, details in status_ckpt_dict.items():
                    self.icons.append(np.array(Image.open(Path(f"{self.ckpts_root}/{details['image']}"))))
                
        ret = []
        for x in range(nodes.shape[0]):
            row = []
            for y in range(nodes.shape[1]):
                icon = np.array(self.icons[nodes[x,y]])
                for idx, osd in enumerate(self.arrows):
                    if roads[x,y] & (1 << idx):
                        icon[osd] = np.array((255,0,0))
                row.append(icon)
                
            ret.append(np.concatenate(row, axis=1))
        ret = np.concatenate(ret, axis=0)
        return ret
    
    def _path_to_direction(self, path: list[Position]) -> float:
        if len(path) <= 1: return None
        curr, next = path[0], path[1]
        pos = next - curr
        match pos:
            case Position(-1, 0):
                return -PI/2
            case Position( 1, 0):
                return  PI/2
            case Position( 0,-1):
                return  PI
            case Position( 0, 1):
                return  0
        return None
    
    def crop_minimap(self, frame: np.ndarray) -> np.ndarray:
        tl = self.region["tl"]
        br = self.region["br"]
        return frame[tl.x: br.x, tl.y: br.y]
    
    def _extract_room(self, minimap: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        grids = np.zeros((5, 5, self.room["size"], self.room["size"], 3), dtype=np.uint8)
        for i in range(5):
            for j in range(5):
                x_start = self.room["tl_pos"].x + i * self.room["interval"]
                y_start = self.room["tl_pos"].x + j * self.room["interval"]
                x_end = x_start + self.room["size"]
                y_end = y_start + self.room["size"]
                grids[j,i] = minimap[y_start:y_end, x_start:x_end]
        
        grids = grids * self.room["mask"]
        hashes = dhash_batch(grids.reshape(25, self.room["size"], self.room["size"], 3))[:,np.newaxis,:]    # [25, 1, 1104]
        distances = np.bitwise_xor(hashes, self.ckpts["hash_vecs"]).sum(axis=2)                             # [25, 11]
        index_map = (distances < self.ckpts["thresh_vecs"]).reshape((5, 5, -1))                             # [5, 5]
        index = np.argmax(index_map, axis=2)
        mask = index_map.any(axis=2)
        return mask.astype(np.uint8) + index, mask
    
    def _get_road_region(self, pos: Position, dir="v"):
        match dir:
            case "v":
                tl = Position(
                    self.room["tl_pos"].x + pos.x * (self.room["interval"]) + self.room["size"],
                    self.room["tl_pos"].y + pos.y * (self.room["interval"]) + self.road["offset"],
                )
                br = tl + Position(self.road["length"], self.road["width"])
                return tl, br
            case "h":
                tl = Position(
                    self.room["tl_pos"].x + pos.x * (self.room["interval"]) + self.road["offset"],
                    self.room["tl_pos"].y + pos.y * (self.room["interval"]) + self.room["size"],
                )
                br = tl + Position(self.road["width"], self.road["length"])
                return tl, br
        raise ValueError("dir must be v or h")
    
    def _extract_road(self, minimap: np.ndarray, grid_mask: np.ndarray):
        X, Y = grid_mask.shape
        vertical   = np.zeros((X, Y), dtype=np.bool_)
        horizontal = np.zeros((X, Y), dtype=np.bool_)
        vertical_pixels, horizontal_pixels = [], []
        for x in range(X):
            for y in range(Y):
                # vertical
                if x + 1 < X:
                    if grid_mask[x+1,y] and grid_mask[x,y]:
                        vertical[x,y] = True
                        tl, br = self._get_road_region(Position(x,y), "v")
                        vertical_pixels.append(minimap[tl.x:br.x, tl.y:br.y])
                # horizontal
                if y + 1 < Y:
                    if grid_mask[x,y] and grid_mask[x,y+1]:
                        horizontal[x,y] = True
                        tl, br = self._get_road_region(Position(x,y), "h")
                        horizontal_pixels.append(minimap[tl.x:br.x, tl.y:br.y])
        
        if len(vertical_pixels) != 0:
            vertical_pixels = np.stack(vertical_pixels).astype(np.uint8)
            mse_v = np.mean(np.abs(vertical_pixels - self.road["color"]), axis=(1,2,3))
            # print(mse_v)
            mse_v = mse_v < 5
        else:
            mse_v = np.zeros(0)
        
        if len(horizontal_pixels) != 0:
            horizontal_pixels = np.stack(horizontal_pixels).astype(np.uint8)
            mse_h = np.mean(np.abs(horizontal_pixels - self.road["color"]), axis=(1,2,3))
            # print(mse_h)
            mse_h = mse_h < 5
        else:
            mse_h = np.zeros(0)
        
        v_cnt, h_cnt = 0, 0
        for x in range(X):
            for y in range(Y):
                if x + 1 < X:
                    if vertical[x,y]:
                        if not mse_v[v_cnt].item():
                            vertical[x,y] = False
                        v_cnt += 1
                if y + 1 < Y:
                    if horizontal[x,y]:
                        if not mse_h[h_cnt].item():
                            horizontal[x,y] = False
                        h_cnt += 1
        
        # 0  1    2    3
        # up down left right
        horizontal = horizontal.astype(np.uint8)
        vertical   = vertical.astype(np.uint8)
        ret = np.roll(vertical,   shift= 1, axis=0) << 0 |   vertical << 1 \
            | np.roll(horizontal, shift= 1, axis=1) << 2 | horizontal << 3
        return ret


class SoulKnightGraph:
    def __init__(self, null_node_idx, node_type_map):
        self.curr_center = Position(0, 0)
        self.null_node_idx = null_node_idx
        self.node_type_map = node_type_map
        self.node_graph = None
        self.access_graph = None
        
        self.last_graph = None
        
    def node_info(self, type_id: int):
        return {
            "type": self.node_type_map[type_id][0],
            "status": self.node_type_map[type_id][1],
            "cost": self.node_type_map[type_id][2],
        }
        
    def is_corrupted(self) -> bool:
        # multiple home, portal, boss in the level
        # portal & boss:
        if self.node_graph is None: return False
        if (self.node_graph == 1).sum() > 1: return True
        tmp = 0
        for i in [12, 13, 14, 15, 16]:
            tmp = self.node_graph == 12 + tmp
        return tmp.sum() > 1
        
    def update(self, update_node: np.ndarray, update_access: np.ndarray) -> tuple[bool, str]:
        if self.is_corrupted():
            print("reset!")
            self.curr_center = Position(0, 0)
            self.node_graph = None
            self.access_graph = None
            self.last_graph = None
        
        if update_node.sum() <= 1:  return (False, "Null Node Graph")
        if update_access.sum() < 1: return (False, "Null Access Graph")
        if self.last_graph is not None:
            if (update_node == self.last_graph).all():
                self.last_graph = update_node
                return (False, "Corrupt Graph")
        self.last_graph = update_node
        
        info = self.node_info(update_node[2, 2])
        if info["type"] is None:
            return (False, "Center Not Detected")
            update_node[2, 2] = 3
            
        top, bottom, left, right = 114514, 0, 1919810, 0
        
        for i in range(update_node.shape[0]):
            for j in range(update_node.shape[1]):
                if update_node[i, j] != self.null_node_idx:
                    top     = min(top,  i)
                    bottom  = max(bottom, i)
                    left    = min(left, j)
                    right   = max(right, j)
                    
        center_pos = Position(2-top, 2-left)
        update_node = update_node[top:bottom+1, left:right+1]
        update_access = update_access[top:bottom+1, left:right+1]
        
        if self.node_graph is None:
            self.curr_center = center_pos
            self.node_graph = update_node
            self.access_graph = update_access
            # print(center_pos, update_node)
            return (True, "Init")
        
        # judge moving direction
        directions = {
            "none":  Position( 0,  0),
            "up":    Position(-1,  0),
            "down":  Position( 1,  0),
            "left":  Position( 0, -1),
            "right": Position( 0,  1),
        }
        

        offset = center_pos - self.curr_center
        direction = None
        for name, pos in directions.items():
            dx, dy = pos.x, pos.y
            legal, flag = False, True
            for i in range(update_node.shape[0]):
                for j in range(update_node.shape[1]):
                    ii = i - offset.x + dx
                    jj = j - offset.y + dy
                    if ii >= self.node_graph.shape[0] or jj >= self.node_graph.shape[1] or ii < 0 or jj < 0:
                        continue
                    if update_node[i, j] == self.null_node_idx or self.node_graph[ii, jj] == self.null_node_idx:
                        continue
                    if self.node_info(update_node[i, j])["type"] != self.node_info(self.node_graph[ii, jj])["type"]:
                        flag = False
                        break
                    else:
                        legal = True
                if not flag:
                    break
            if legal and flag:
                direction = name
            
        if direction is None or direction == "none":
            return (False, "Null Direction")

        or_x, or_y = self.node_graph.shape
        or_c_x, or_c_y = self.curr_center.x, self.curr_center.y
        
        up_x, up_y = update_node.shape
        up_c_x, up_c_y = center_pos.x, center_pos.y
        
        or_x_u, or_y_l = or_c_x, or_c_y
        or_x_d, or_y_r = or_x - or_c_x, or_y - or_c_y
        
        up_x_u, up_y_l = up_c_x, up_c_y
        up_x_d, up_y_r = up_x - up_c_x, up_y - up_c_y
        
        # print(max(or_x_u, up_x_u - directions[direction][0]), max(or_x_d, up_x_d - directions[direction][1]))
        origin_graph_center = Position(
            max(or_x_u, up_x_u - directions[direction].x),
            max(or_y_l, up_y_l - directions[direction].y)
        )
        
        update_node_center = origin_graph_center + directions[direction]
        
        new_graph_shape = Position(
            origin_graph_center.x + max(or_x_d, up_x_d + directions[direction].x),
            origin_graph_center.y + max(or_y_r, up_y_r + directions[direction].y)
        )
        
        origin_graph_lu = Position(origin_graph_center.x - or_x_u, origin_graph_center.y - or_y_l)
        origin_graph_rd = Position(origin_graph_center.x + or_x_d, origin_graph_center.y + or_y_r)
        
        update_node_lu = Position(update_node_center.x - up_x_u, update_node_center.y - up_y_l)
        update_node_rd = Position(update_node_center.x + up_x_d, update_node_center.y + up_y_r)
        
        new_node    = np.zeros((new_graph_shape.x, new_graph_shape.y), dtype=np.int8)
        new_access  = np.zeros((new_graph_shape.x, new_graph_shape.y), dtype=np.int8)
        for or_x, new_x in enumerate(range(origin_graph_lu.x, origin_graph_rd.x)):
            for or_y, new_y in enumerate(range(origin_graph_lu.y, origin_graph_rd.y)):
                if self.node_graph[or_x, or_y] != self.null_node_idx:
                    new_node[new_x, new_y] = self.node_graph[or_x, or_y]
                    new_access[new_x, new_y] |= self.access_graph[or_x, or_y]
                # if self.access_graph[or_x, or_y] != 0:
        
        for up_x, new_x in enumerate(range(update_node_lu.x, update_node_rd.x)):
            for up_y, new_y in enumerate(range(update_node_lu.y, update_node_rd.y)):
                if update_node[up_x, up_y] != self.null_node_idx:
                    new_node[new_x, new_y] = update_node[up_x, up_y]
                    new_access[new_x, new_y] |= update_access[up_x, up_y]
                # if self.access_graph[or_x, or_y] != 0:
        
        self.node_graph     = new_node
        self.access_graph   = new_access
        self.curr_center    = update_node_center
        
        return (True, direction)


    def find_path(self, target_type, target_status):
        if self.access_graph is None or self.node_graph is None: return None
        rows, cols = self.node_graph.shape[0], self.node_graph.shape[1]
        start_x, start_y = self.curr_center.x, self.curr_center.y

        # 检查起始点是否已经是目标节点
        start_type_id = self.node_graph[start_x][start_y]
        start_info = self.node_info(start_type_id)
        if start_info['type'] == target_type and start_info['status'] == target_status:
            return [Position(start_x, start_y)]
        # 初始化优先队列，堆中的元素是 (累计代价, x坐标, y坐标)
        heap = []
        heappush(heap, (start_info['cost'], start_x, start_y))

        # 记录到达每个节点的最小代价
        cost_so_far = np.full((rows, cols), np.inf)
        cost_so_far[start_x][start_y] = start_info['cost']

        # 记录前驱节点，用于回溯路径
        came_from = {}

        # 定义四个方向的位移及对应的访问掩码
        directions = [
            (-1, 0, 1),   # 上: 掩码1
            (1, 0, 2),    # 下: 掩码2
            (0, -1, 4),   # 左: 掩码4
            (0, 1, 8)     # 右: 掩码8
        ]

        found = False
        result = None

        while heap:
            current_cost, x, y = heappop(heap)

            # 如果当前节点的代价已经大于记录的最小代价，跳过
            if current_cost > cost_so_far[x][y]:
                continue

            # 检查当前节点是否是目标节点
            current_type_id = self.node_graph[x][y]
            current_info = self.node_info(current_type_id)
            
            if current_info['type'] == target_type and current_info['status'] == target_status:
                # 构造路径
                path = []
                current_pos = (x, y)
                while current_pos is not None:
                    px, py = current_pos
                    path.append(Position(px, py))
                    # path.append((px, py))
                    current_pos = came_from.get(current_pos, None)
                path.reverse()
                result = path
                found = True
                break

            # 遍历四个方向
            for dx, dy, mask in directions:
                # 检查是否有访问权限
                if not (self.access_graph[x][y] & mask):
                    continue
                nx = x + dx
                ny = y + dy
                # 检查坐标是否合法
                if nx < 0 or nx >= rows or ny < 0 or ny >= cols:
                    continue
                # 检查相邻节点是否存在
                if self.node_graph[nx][ny] == self.null_node_idx:
                    continue
                # 计算新代价
                neighbor_type_id = self.node_graph[nx][ny]
                neighbor_info = self.node_info(neighbor_type_id)
                neighbor_cost = neighbor_info['cost']
                new_cost = current_cost + neighbor_cost
                # 如果找到更优的路径
                if new_cost < cost_so_far[nx][ny]:
                    cost_so_far[nx][ny] = new_cost
                    came_from[(nx, ny)] = (x, y)
                    heappush(heap, (new_cost, nx, ny))

        return result if found else None

if __name__ == "__main__":
    import json, time
    import cv2
    from client import Client
    
    config = json.load(open("./configs/minimap.json"))
    
    minimap = SoulKnightMinimap(config)
    # mini_map_region = (1060, 120, 1260, 320)
    # frame = np.fromfile("./res/brighter.bin", dtype=np.uint8).reshape(720, 1280, 4)[:, :, :3][mini_map_region[1]:mini_map_region[3], mini_map_region[0]:mini_map_region[2]]
    # # Image.fromarray(frame).show()
    # update_node, update_access = minimap.parse(frame)
    # graph.update(update_node, update_access)
    # img = render_minimap(graph.node_graph, graph.access_graph)
    # img.show()
    
    mini_map_region = (1060, 120, 1260, 320)
    client = Client("SKM_16448", ip="127.0.0.1", port="55555", timeout=1)
    while True:
        frame = client.fetch_fb()
        frame = frame[mini_map_region[1]:mini_map_region[3], mini_map_region[0]:mini_map_region[2]]
        try:
            print("GoooooooogleMap: ", minimap.navigate())
            dir = minimap.update(frame)
            print(f"current at: {minimap.graph.curr_center}")
            print("node:")
            print(minimap.graph.node_graph)
            print("access:")
            print(minimap.graph.access_graph)
            img = minimap.render()
            if img is not None:
                cv2.imshow("minimap", np.array(img))
                key = cv2.waitKey(100)
        except ValueError as e:
            pass
        finally:
            time.sleep(0.5)
