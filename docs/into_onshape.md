# Getting the data into onshape

[Forum](https://forum.onshape.com)

## 2 D sketch

Xtrkcad can export the plan to a dxf file, which can be imported into onshape.  This effectively brings in the x,y coordinates of the track junctions.  

Note: turn off the labels under options -> display -> enable labels.  Select all tracks and all the table edges. Then File -> Export DXF.

Import: 
- create new onshape file
- at bottom left click on the + sign, then import
- create a new sketch and click on the the DXF import (top icon bar)

The X and Y coordinates are changed during this process.

## x,y calibration

The x and y coordinates are transformed during the process.  The origin is moved to somewhere in the middle of the diagram (not exactly the center).  The strategy is offset the data points so that they are identical to what onshape has.

In onshape - go to the dxf tab (which renders the dxf you imported). There is a tool set in the upper left. Use the coordinates tool to get the values for a point

  X = 26.3750 Y = 0.5000 Z = 0.0000

Using one of the table edges (polyline in my case), for instance, you can confirm the dxf values by inspecting the .xtc file:

```bash
DRAW 143 6 0 0 0 0.000000 0.000000 0 0.000000
	Y4 3050200 0.000000 4 2 
		2.468959 9.484067 0
		26.343541 9.625183 0
		26.375000 0.500000 0
		202.127216 0.647419 0
	END$SEGS
  ```
On Parts Studio tab get the same points values

  Point: X -88.417 Y -52.008 Z 0.000 in

Calculate adjusters for x and y
new - orig = adj
x: -88.417 - 26.375 = -114.792
y: -52.008 -0.5 = -52.508

Test this with another point

Original endpoint on curved track as shown on xtrkcad ui:
49.635, 18.898
+ adj is
-65.157, -33.61

UI on onshape yields
-65.158 , -33.609

This indicates a rounding error in the thousandths.

The xtc file has more places

49.634974, 18.898439

Going back to original point shows no more significant places
26.375000 0.500000

Hovering over the point values in the lower right of onshape shows 6 places. Also clicking on the measurement details icon and then clicking on the x and y values displays all 6 places.

-88.417500, -52.007813

Revised adjusters are:
-114.7925, -52.507813

Adjusting the test point
-65.157526, -33.609374

On shape shows
-65.157526, -33.609373

Thus we need to use all 6 places and use fuzzy matching at the 6th place to locate the position.  BTW the track end point is the center of the track at the end point.

## Heights

A csv file is the exchange mechanism to get the heights into onshape

Onshape can read the csv file with their Feature Script.

Add Table (Pascoe) to tool bar as shown [here](https://forum.onshape.com/discussion/19737/table-new-custom-feature)

Or pull the table directly with FS as described [here](https://cad.onshape.com/FsDoc/imports.html). Use the icon box with down arrow to create the following line

```
import(path : "e02886384a679df24211a12b", version : "e9a090a56c0ed527011a1570");
```

This makes BLOB_DATA available

Read the data such as:

```
for (var row in BLOB_DATA.csvData)
  {
    for (var cell in row)
    {
      print(cell ~ ', ');
    }
    println('');
  }
```

### Queries

Another tool is Query Explorer - add that too.
Setting that to NOTICES will put the query in the FeatureScript notices panel

For instance all arcs is

qGeometry(qEverything(), GeometryType.ARC)

All lines is: 

qGeometry(qEverything(), GeometryType.LINE)

qUnion([array of queries]) -- its an OR
qIntersection ([array of queries]) -- its AND

There is not geometry type for point.  That is done with

qEverything(EntityType.VERTEX)

The lines include the lines that extend past the guage at the junctions of parts.

The arcs and lines don't necessarily correspond to the the track parts.  They may correspond to the internal paths on switches.

This looks promising

qGeometry(qContainsPoint(qEverything(), vector(-56.87678, -42.02087, 0) * inch), GeometryType.LINE)

Oddly these seem to resolve to double (?) the expected number of edges:

Hello world!debug: Query resolves to 8 edges
▼▼▼▼ Final query ▼▼▼▼
qContainsPoint(qEverything(EntityType.EDGE), vector(-39.66672, -49.97732, 0) * inch


evaluateQuery (context is Context, query is Query) returns array

Returns an array of queries for the individual entities in a context which match a specified query. The returned array contains exactly one transient query for each matching entity at the time of the call. If the context is modified, the returned queries may become invalid and no longer match an entity.

It is usually not necessary to evaluate queries, since operation and evaluation functions can accept non-evaluated queries. Rather, the evaluated queries can be used to count the number of entities (if any) that match a query, or to iterate through the list to process entities individually.

const lines = evaluateQuery(context, lineQuery);
for (var line in lines)
{
    const lineLength = evLength(context, { "entities" : line });
    // extrude(...) or opExtrude(...) using lineLength
}