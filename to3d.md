STRATEGY
Use the elevations.py program to create a spreadsheet of all elevations.

## Pillars

Select points where tracks meet along the route and raise "pillars" to "support the 3d route.
Extrude half of the crosstie at the each track junction to height established by xtrkcad. 
Name these Pillar T*n*, where T*n* is the identifier used by xtrkcad.
The inner corner then is a point on the centerline of the roadbed as it rises.
If the cross-tie is the start of tunnel it will be the whole tie, but it will have a center point.

## Centerline of roadbed

The centerline is a 3d curved line that is smooth and continuous, supporting a sweep.
Create anchors of either straights or helicies, then connect them with bridging curves.

### Anchor on straights
Each straightaway connects these points with a trivial 3d spline (labeled as 3d strait 1,2,3)

### Anchor with helix

If there is too much distance between straights and variation of the curve it won't work right.
In that case create another anchor for the bridging curve using a helix.

Create the circular edge by extruding the center rail to the height of the lower side of the curve.
Name this extrusion Helix *n* curve
Helix is constructed with the height and turns method. 
Clockwise seems to be the right selection.
Note the angular length from xtrkcad.  Divde by 360 to get the number of revolutions.
Used the edge of the extruded center rail as the cylindrical surface
The start angle is determined by trail and error. The line must start and end on the corners of the "pillar" extrusions

### Bridging curves

Join the anchors with bridging curves.  
Its best to connect both sides of the helix with bridging curves - otherwise the line won't be smooth and can't support a sweep.
Select vertex of the anchor for the start and end. 
Look straigt down and used the controls to match the center line from the dxf.

### Grouping sections together

For the sweep - use the composite curve tool to group sections of track instead of enumerating each one in the sweep.