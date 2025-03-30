import cv2
import numpy as np
from utils.position import Position

from sklearn.cluster import DBSCAN

def apply_filter(img: np.ndarray, filters: dict[tuple[Position, Position]], value=0) -> np.ndarray:
    for tl, br in filters.values():
        img[tl.x:br.x, tl.y:br.y] = value
    return img

def obj_detect(fb) -> np.ndarray:
    # origin = cv2.cvtColor(fb.copy(), cv2.COLOR_RGB2BGR)
    H, W, C = fb.shape
    ret = np.zeros((H//4, W//4), dtype=np.uint8)
    
    edges = (fb.mean(axis=2) < 5).astype(np.uint8) * 255
    edges = apply_filter(edges, {
        "level_num":  (Position(290, 1060), Position(320, 1130)),
        "level_desp": (Position(115, 1110), Position(145, 1210)),
        "coin&stop":  (Position( 20, 1030), Position(100, 1270)),
        "health":     (Position(  0,    0), Position(128,  250)),
    })
    
    points = np.argwhere(edges > 0)[:, [1, 0]].astype(np.float32)
    if len(points) == 0: return ret
    
    dbscan = DBSCAN(eps=3, min_samples=5)
    labels = dbscan.fit_predict(points)
    
    contours = []
    for i in range(np.max(labels) + 1):
        mask = labels == i
        if np.sum(mask) < 100: continue
        contours.append(points[mask])
    
    for contour in contours:
        if len(contour) < 200: continue
        hull = cv2.convexHull(contour).astype(np.int32)
        ret = cv2.fillConvexPoly(ret, hull//4, 1)
        
    return ret

if __name__ == "__main__":
    import time
    from client import Client
    
    client = Client("SKM_16448", ip="127.0.0.1", port="55555", timeout=1)
    # fb = client.fetch_fb()
    # start = time.perf_counter()
    # for i in range(200): 
    #     print(i)
    #     obj_detect(fb)
    # end = time.perf_counter()
    # print(f"{round((end-start) * 1000 / 200, 2)}ms")
    # exit()
    
    while True:
        fb = client.fetch_fb()
        try:
            start = time.perf_counter()
            frame = obj_detect(fb)
            print(f"{round((time.perf_counter() - start) * 1000, 2)}ms")
            cv2.imshow('Clustered Edges', frame)
            cv2.waitKey(5)
            
            # centers = obj_detect(fb.copy())
            # fb = cv2.cvtColor(fb, cv2.COLOR_RGB2BGR)
            # fb = cv2.resize(fb, (fb.shape[1] // 4, fb.shape[0] // 4))
            # for center in centers:
            #     x, y = center
            #     cv2.circle(fb, (x, y), 5, (0, 0, 255), -1)
            # cv2.imshow('Clustered Edges', fb)
            # cv2.waitKey(5000)

            # objects = obj_detect(fb.copy())
            # fb = cv2.cvtColor(fb, cv2.COLOR_RGB2BGR)
            # fb = cv2.resize(fb, (fb.shape[1] // 4, fb.shape[0] // 4))
            # for obj in objects:
            #     tl, br = obj
            #     print(tl, br)
            #     fb = cv2.rectangle(fb, (tl.x, tl.y), (br.x, br.y), (0, 255, 0), 2)
                
            # print()
            # cv2.imshow('Clustered Edges', fb)
            # cv2.waitKey(500)
        except Exception as e:
            print(e)
            raise e
        finally:
            cv2.destroyAllWindows()