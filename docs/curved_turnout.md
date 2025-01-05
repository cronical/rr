# Curved Turnout Definition

## Background

I wanted to use curved turnouts on the new plan and selected one from the parameter file library.  It turned out that this model was no longer available.  The vendor does not provide an XTrackCAD definition, only a pdf. I attempted to create the definition from that, some some, but not complete success.  I ended printing out the definition of 849-83602 and physcially laying it over the correctly scaled 948-83602 and decided it was close enough.  Below is some info about what I tried in case I ever come back to this.

## Tour of Geometry

The basic plan of the switch is a short straight segment at each end.  Both the inner and out legs connect their end segments to the starting straight segment via a curve of constant radius.  

The program `build_curve_turnout_def` defines the key values in a dictionary/list format.  It turns this into the format used by the parameter files of XTrackCAD.  There are problems.

1. Unresolved problem with the mirrored right and turnout, which appears to be reading the segments in the wrong order.
1. The mirrored right turnout apparently works only when the first angle field is set to zero.  The unmirrored one does not have this issue.
1. The values provided actually over specifify the geometry, so if they don't match it creates a not connecting error on loading.  This is because I specify two tangent line segments AND the radius.  The radius should be computable from the line segments, knowing that they are tangent.  When I tried this, though, I encountered a problem because I could not actually determine with certainty the location and orientation of the line segments from the vendor's pdf.



