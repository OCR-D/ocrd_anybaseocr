import json
import os
from collections import defaultdict

annotations = json.load(open(os.path.join("", "data.json")))
#annotations = list(annotations.values())  # don't need the dict keys
#print(json.dumps(annotations, indent=2))
# The VIA tool saves images in the JSON even if they don't have any
# annotations. Skip unannotated images.
#annotations = [a for a in annotations if a['regions']]

# Add images
# Get the x, y coordinaets of points of the polygons that make up
# the outline of each object instance. These are stores in the
# shape_attributes (see json format above)
# The if condition is needed to support VIA versions 1.x and 2.x.
polygons = defaultdict(list)
shape = list()
for a in annotations:
	polygons.clear()
	image_path = os.path.join("block/train/", a,".tif")
	for x in annotations[a]:
		if x['block_class'] not in shape:
			shape.append(x['block_class'])
		polygons[a].append({
			'name': x['block_class'],
			'all_x_values': [r for r in x['all_x_values']],
			'all_y_values': [r for r in x['all_y_values']]
		})
print((shape))

