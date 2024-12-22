from .region import RegionBase, CircleRegion, RectRegion

def get_region(region_type: str, **kwargs) -> RegionBase:
    region_type = region_type.strip().lower()
    match region_type:
        case "circle":
            return CircleRegion(**kwargs)
        case "rect":
            return RectRegion(**kwargs)
        case _:
            raise ValueError(f"Unknown region type: {region_type}")
