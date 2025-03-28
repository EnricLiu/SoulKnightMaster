import cv2 as cv
import cv2
import numpy as np
from PIL import Image

from position import Position

from sklearn.cluster import DBSCAN


def apply_filter(img, filters: dict[tuple[Position, Position]]) -> np.ndarray:
    for _, f in filters.items():
        tl, br = f
        img[tl.x:br.x, tl.y:br.y] = 0
    return img

def obj_detect(fb):
    # origin = cv2.cvtColor(fb.copy(), cv2.COLOR_RGB2BGR)
    
    edges = (fb.mean(axis=2) < 4).astype(np.uint8) * 255
    edges = apply_filter(edges, {
        "level_num":  (Position(290, 1060), Position(320, 1130)),
        "level_desp": (Position(115, 1110), Position(145, 1210)),
        "coin&stop":  (Position( 20, 1030), Position(100, 1270)),
        "health":     (Position(  0,    0), Position(128,  250)),
    })
    
    points = np.argwhere(edges > 0)[:, [1, 0]].astype(np.float32)
    
    dbscan = DBSCAN(eps=3, min_samples=5)
    labels = dbscan.fit_predict(points)
    
    contours = []
    for i in range(np.max(labels) + 1):
        mask = labels == i
        if np.sum(mask) < 200: continue
        contours.append(points[mask])
    
    objects = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        merged_x, merged_y, merged_w, merged_h = x, y, w, h
        to_remove = []

        for obj in objects:
            obj_x, obj_y, obj_w, obj_h = obj['bounding_box']
            
            if not (merged_x + merged_w < obj_x or 
                    merged_x > obj_x + obj_w or 
                    merged_y + merged_h < obj_y or 
                    merged_y > obj_y + obj_h):
                
                new_x = min(merged_x, obj_x)
                new_y = min(merged_y, obj_y)
                new_right = max(merged_x+merged_w, obj_x+obj_w)
                new_bottom = max(merged_y+merged_h, obj_y+obj_h)
                
                merged_w = new_right - new_x
                merged_h = new_bottom - new_y
                merged_x, merged_y = new_x, new_y
                
                to_remove.append(obj)

        for obj in to_remove:
            objects.remove(obj)

        objects.append({
            'size': (merged_w, merged_h),
            'bounding_box': (merged_x, merged_y, merged_w, merged_h),
        })
        
    return objects

if __name__ == "__main__":
    import time
    from utils import Action
    from train.client import Client
    
    client = Client("SKM_16448", ip="127.0.0.1", port="55555", timeout=1)
    # fb = client.fetch_fb()
    
    # start = time.perf_counter()
    # for i in range(200): obj_detect(fb)
    # end = time.perf_counter()
    # print(f"{round((end-start) * 1000 / 200, 2)}ms")
    # exit()
    
    while True:
        fb = client.fetch_fb()
        try:
            objects = obj_detect(fb.copy())
            fb = cv2.cvtColor(fb, cv2.COLOR_RGB2BGR)
            for obj in objects:
                x, y, w, h = obj['bounding_box']
                fb = cv2.rectangle(fb, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv.imshow('Clustered Edges', fb)
            cv.waitKey(50)
        except Exception as e:
            print(e)
        finally:
            cv2.destroyAllWindows()