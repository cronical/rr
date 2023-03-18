'''The part class defintion'''
from math import asin, degrees, pi
import numpy as np
import pandas as pd

# formatters
def decorate(df) -> pd.DataFrame:
  '''Put the T for track on the part numbers'''
  for fld in ['part_index','connects_to']:
    df[fld]='T' + df[fld].astype(str)
  return df
def fmt_node_id(part_a,connects_to=None,end_ref=None):
  '''create string node-id 
  argument: 
    The part object that is connecting and either 
    connects_to: another part object or
    end_ref: a sequential number to distinguish all between the items that are not connected 
  returns: a string with the lower valid part number 1st followed by a hyphen and the higher part number,
    unless there is a negative, in which case the unconnected item comes 2nd
    Decorators are used to make them node name more readable:
      C curve
      E an unconnected node (ends)
      S straight
      T turnout
  examples:
    S95-T99 - The place where parts straight 86 and turnout 97 connect.
    C101-E5 - A curve that ends
  
  assumes that two tracks never connect to each other twice
  '''
  assert (connects_to is None) != (end_ref is None),'Exactly one of connects_to and end_ref should be provided'
  a=str(part_a)
  decorators=[a[0]]
  ids=[int(a[1:])]
  if connects_to is not None:
    b=str(connects_to)
    decorators+=[b[0]]
    ids+=[int(b[1:])]
    a=sorted(list(zip(ids,decorators)))
  else:
    decorators+=['E']
    ids+=[end_ref]
    a=list(zip(ids,decorators))
  r='-'.join([t+str(i)for i,t in a])
  return r 
def fmt_point(point)->str:
  '''format pairs as points'''
  x,y=point
  return '(%.6f, %.6f)'%(x,y)

# geometry functions
def get_angle(a, b, c):
  '''Angular length for helix.  b is the center, a and c are the wings'''
  opposite=get_length(a,c)*.5
  radius=get_length(a,b)
  radians=2*abs(asin(opposite/radius))
  ang = degrees(radians)
  return ang
def get_length(point1,point2):
  '''distance between two points'''
  point1=np.array(point1)
  point2=np.array(point2)
  distance=(np.subtract(point1,point2)**2).sum()**.5
  distance=round(distance,6)
  return distance
  #self.length = ((x1 - x2)**2 + (y1 - y2)**2)**0.5

class Part:
  '''represents a piece of track with end points, and paths'''
  part_types=('STRAIGHT','CURVE','TURNOUT')
  def __init__(self, id ,part_type):
    self.id= id
    self.part_type=part_type
    assert part_type in Part.part_types
    self.end_points=[]# list of [connects_to, x,y,z,angle]
    # z is the defined elevation if provided, otherwise zero until it is calculated
    # which can only be done based on surrounding parts' defined elevations
    self.length=0
    self.mfg_info=None

    # for curves
    self.center=None
    self.angular_length=None
    self.revs=None # revolutions for helix

    # for Turnouts
    self.segments=[] # in the form ((x1,y1),(x2,y2))
    # only for turnouts where there may will be more than 1
    self.paths=[] # a list of dicts with keys: "text","steps","length","divergence"
    # step elements are indices into the segments origin 1 (as they are in xtrkcad)
    # for left empty for straights and curves
    self.group=0
    self.turnout_orig=(0,0,0)
    self.turnout_angle=0
  
  def __str__(self):
    '''return code + part number where code S=straight, C=curve, T=Turnout'''
    return '%s%d'%(self.part_type[0:1],self.id)  

  def __repr__(self):
    str='%s %d'%(self.part_type,self.id)
    if self.mfg_info:
      str+=' (%s)'%(' '.join(self.mfg_info))
    str+='\n'
    if self.turnout_orig:
      x,y,z=self.turnout_orig 
      str+='\torig: %.6f, %.6f, %.6f angle: %.6f\n'%(x,y,z,self.turnout_angle)
    for ep in self.end_points:
      c,x,y,z,angle=ep
      if c is None:
        ct='None'
      else:
        ct='%d'%c
      str+= '\tConnects to: %s at coordinates (%.6f, %.6f, %.6f) on angle %.6f\n'%(ct,x,y,z,angle)
    match self.part_type:
      case 'CURVE':
        x,y=self.center
        str+='\tCenter: (%.6f, %.6f)\n '% (x,y)
        str+='\tLength: %.6f\n '%(self.length)
        str+='\tAngular Length: %.6f\n '%(self.angular_length)
        str+='\tRevolutions: %.6f\n '%(self.revs)
      case 'STRAIGHT':
        str+='\tLength: %.6f\n '%(self.length)
      case 'TURNOUT':
        pass
        for path in self.paths:
          p=' '.join('%d'% a for a in path['steps'])
          str+= '\tPath: %s: %s (length: %.6f)\n '%(path['text'],p,path['length'])
        for segment in self.segments:
          match segment['type']:
            case 'S':
              s=' '.join(fmt_point(segment[a]) for a in ['point1','point2'])
              str+= '\t%s points: %s\n '%(segment['type'],s)
            case 'C':
              a='center'
              c='%s: %s'%(a,fmt_point(segment[a]))
              s=['%s: %.6f'%(a,segment[a]) for a in ['radius','angle','swing']]
              s.insert(1,c)
              str+= '\t%s points: %s\n '%(segment['type'],' '.join(s))
          
    return str

  def set_center(self,x,y):
    '''for curve set center'''
    self.center=[x,y  ]

  def add_end_point(self,connects_to,x,y,elev_defined,angle): # connects_to is None for E4
    self.end_points+=[[connects_to,x,y,elev_defined,angle]]
    match self.part_type:
      case 'STRAIGHT':
        if len(self.end_points)==2:
          self.length=get_length(self.end_points[0][1:3],self.end_points[1][1:3])
      case 'CURVE':
        if len(self.end_points)==2:
          self.angular_length=get_angle(a=self.end_points[0][1:3],b=self.center,c=self.end_points[1][1:3])
          radius=get_length(self.end_points[0][1:3],self.center)
          self.length=round((self.angular_length/360)* 2* pi * radius,6)
          self.revs=self.angular_length/360
          pass
      case 'TURNOUT':
        pass

  def add_straight_segment(self,point1,point2):
    '''for turnouts only, segment type S
    point1 and point2 are x,y pairs'''
    assert self.part_type=='TURNOUT'
    segment={'type':'S','point1':point1,'point2':point2}
    self.segments+=[segment]

  def add_curve_segment(self,radius,center,angle,swing):
    '''for turnouts only, segment type C or C3 ?
    center is an x,y pair'''
    assert self.part_type=='TURNOUT'
    segment={'type':'C','radius':radius,'center':center,'angle':angle,'swing':swing}
    self.segments+=[segment]

  def add_path(self,label,steps):
    '''Add a path through the part
    steps is a list of segment numbers origin 1'''
    assert self.part_type=='TURNOUT'
    path={ 'text':label,'steps':steps,'length':0}
    self.paths+=[path]
  
  def  validate_turnout(self):
    '''once all the paths and segments are input, validate and calcuate length and divergence'''
    assert self.part_type=='TURNOUT'
    for path in self.paths:    
      path_len=[]
      step_divergence=[]
      for sn in path['steps']:
        seg=self.segments[sn-1]
        match seg['type']: # get the length for all segments along path
          case 'S':
            step_end_point=seg['point2']
            path_len+=[abs(get_length(seg['point1'],step_end_point))]
            step_divergence+=[0]
          case 'C': # in the case of curves also alter the angle
            path_len+=[(seg['swing']/360)* 2* pi * abs(seg['radius'])]
            step_divergence+=[seg['swing']*np.sign(seg['radius'])]
        path['length']=round(sum(path_len),6)
        assert path['length'] !=np.nan, 'length is nan'
        assert path['length'] >0, 'length is zero!'
        path['divergence']=sum(step_divergence)

  def has_connection_to(self,id):
    return id in [a[0] for a in self.end_points]

  def mono_to_connection(self,conn_id):
    '''from the first connection to this one return true if height increases or stays the same'''
    h=[a[1]for a in self.end_points if a[0]==conn_id]
    return h[0] >= self.end_points[0][1]
