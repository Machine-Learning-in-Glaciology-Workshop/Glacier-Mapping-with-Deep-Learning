def region_filter(regions):
    def filter(tile_group):
        return (tile_group.name[1:].split('-')[0] in regions)
    return filter
