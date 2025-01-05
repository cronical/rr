#!/usr/bin/env python3
from math import sin, cos, acos, radians,degrees
from operator import itemgetter
import numpy as np
from part_class import Part,add_points,get_length,pi
"""Create left and right curved turnout definition suitable for a parameter file.
There are 
- 3 end points, each has a short straight segment.
- 2 paths of length 3, sharing the starting short straight segment
- 3 straight and 2 curved segments
"""

def locate_curve_end(center_end:tuple,straight_len:float,tan_angle:float)->tuple:
  """locate the joint between the curve and the terminal straight section.
  center_end is an x,y tuple
  straight_len is the length of the terminal straight section
  tan_angle is the angle of the tangent away from the end point in degrees
  
  result is an x,y pair where the curve ends and the straight section begins.
  """
  theta = radians(tan_angle)
  x=-straight_len * cos(theta)
  y=-straight_len * sin(theta)
  if tan_angle>90:
    y=-y
  result=add_points(center_end,(x,y))
  return result

def get_angle_opposite_c(a:float,b:float,c:float)->float:
  """ For triange with sides of length, a,b,c 
  use the law of cosines to get the angle opposite of side of length c.
  Return the angle in degrees.
  """
  gamma=acos(((a**2.0)+(b**2.0)-(c**2.0))/ (2.0*a*b))
  return degrees(gamma)
fld_info="""
each leg has a list of segments:
seg_end is the location of the end of the center line for each leg
  these are positive and used as such for the left orientation.
  program negates the y value (only) for the right orientation.

CURVES:
radius - stated as positive, but negated for left orientation - curved segments only

STRAIGHTS:
straight_len is the length of the straight segment at the end of the leg - straight segments only
tan_angle is direction of the tangent (away) at the end point (on the left version)

"""

values={
  "mfg":"Walthers DCC",
  "descr": "20-24 %s Hand Curve Turnout", # will set to left or right
  "initial":{
    "segments":[
      {
        "seg_end": (.612,0),
        "tan_angle": 270,
        "straight_len": .65625
      }
    ]
  },
  "inner":{
    "segments":[
      {
      "seg_end":(11.937,4.441),
      "radius": 16.668
      },
      {
      "seg_end":(13.13,5.534),
      "tan_angle":48.1229,
      "straight_len":1.581,
      }
    ]
  },
  "outer":{
    "segments":[
      {
        "seg_end":(13.092,3.965),
        "radius": 21.7  
      },
      {
        "seg_end":(13.324,4.149),
        "tan_angle":53.269,
        "straight_len": .612,
      }
    ]

  }
}
orientation=zip(["left","right"],[1,-1],[-1,1],["948-83061","948-83062"])

def output_as_parameter(part):
  """Formats a turnout part and send to stdout
  """
  assert part.part_type=="TURNOUT"
  title='\t'.join( part.mfg_info)
  print('TURNOUT HO '+f'"{title}"')
  for path in part.paths:
    s= ' '.join(["%d"%a for a in path['steps']])
    text=path['text']
    print(f'\tP "{text}" '+s)
  for end_point in part.end_points:
    # order believed not to matter
    ep=["E"]+["{:.6f}".format(a) for a in itemgetter(1,2,4)(end_point)]
    print ("\t"+" ".join(ep))
  for segment in part.segments:
    match segment['type']:
      case 'S':
        floats=[0]+list(segment['point1'])+list(segment['point2'])
        s=["S","0"]+["{:.6f}".format( a) for a in floats]
        print("\t"+" ".join(s))
      case 'C':
        sa=segment["angle"]
        if part.mfg_info[2].endswith('83062'):
          sa=0
        floats=[0]+[segment["radius"]]+list(segment["center"])+[sa,180-segment["angle"]]
        c=["C","0"]+["{:.6f}".format( a) for a in floats]
        print("\t"+" ".join(c))

  print ("END$SEGS")
    

def main():
  print ("CONTENTS Walthers Track HO DCC Code 83 Turnouts by Dobbs")
  print ("SUBCONTENTS Walthers Track HO DCC Code 83 - Curved Turnouts")

  for part_no,(hand,y_flip,r_flip,model_no) in enumerate(orientation):
    #initial segment
    part=Part(part_no,"TURNOUT")

    descr=values["descr"]%hand
    part.mfg_info= values["mfg"],descr,model_no

    seg_start=(0,0)
    for leg in 'initial','inner','outer':
      _path=[1]

      # gather the values
      # there will be either one segment in the leg (initial) or two (a curve and a straight)
      radius=tan_angle=curve_end=straight_end=None
      for segment in values[leg]["segments"]:
        match 'radius' in segment:
          case False: # straight
            tan_angle=segment["tan_angle"]
            straight_end=tuple(np.array(segment["seg_end"])*np.array((1,y_flip)))
          case True: #curve
            radius=segment['radius']
            curve_end=tuple(np.array(segment["seg_end"])*np.array((1,y_flip)))
            pass

      if radius is not None:
        center=add_points(seg_start,(0,y_flip*radius))

        # distance=get_length(seg_start,seg_end) # length of chord of curve
        # gamma=get_angle_opposite_c(radius,radius,distance) # angle inscribed by arc

        # The tan_angle of the final segment is the same as the final angle of the curve
        # apparently, we use the supplement of the curve's angle 
        part.add_curve_segment((radius*r_flip),center,180-tan_angle,0)
        _path.append(len(part.segments)) 
        seg_start=curve_end
        pass

      part.add_straight_segment(seg_start,straight_end) 
      seg_start=straight_end 
      match leg=='initial':
        case False:
          x,y=straight_end
          part.add_end_point(0,x,y,0,90 + (r_flip*tan_angle))
        case True:
          part.add_end_point(0,0,0,0,tan_angle)
      _path.append(len(part.segments)) 
      pass
      if leg=='initial':
        continue
      part.add_path(leg,_path)

    output_as_parameter(part)
  pass


if __name__=="__main__":
  main()