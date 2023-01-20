#!/usr/bin/env python3
'''Read connected track defined and computed elevations out of the xtrkcad file and save into a csv'''
import logging
import math
from os import path
import re
import yaml
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO)

def get_length(point1,point2):
  '''distance between two points'''
  point1=np.array(point1)
  point2=np.array(point2)
  distance=(np.subtract(point1,point2)**2).sum()**.5
  return distance
  #self.length = ((x1 - x2)**2 + (y1 - y2)**2)**0.5

def fmt_point(point)->str:
  '''format pairs as points'''
  x,y=point
  return '(%.6f, %.6f)'%(x,y)

class Part:
  '''represents a piece of track with end points, and paths'''
  part_types=('STRAIGHT','CURVE','TURNOUT')
  def __init__(self, id ,part_type):
    self.id= id
    self.part_type=part_type
    assert part_type in Part.part_types
    self.end_points=[]# list of [connects_to, x,y,z]
    # z is the defined elevation if provided, otherwise zero until it is calculated
    # which can only be done based on surrounding parts' defined elevations
    self.segments=[] # in the form ((x1,y1),(x2,y2))
    # only for turnouts where there may will be more than 1
    self.paths=[] # a list of dicts with keys: "text","steps","length"
    # stepselements are indices into the segments origin 1 (as they are in xtrkcad)
    # for left empty for straights and curves
    self.group=0
    self.length=0
    self.center=None
    self.angular_length=None
    self.turnout_detail=None
    self.revs=None # revolutions for helix
  def __repr__(self):
    str='%s %d\n'%(self.part_type,self.id)
    for ep in self.end_points:
      c,x,y,z=ep
      if c is None:
        ct='None'
      else:
        ct='%d'%c
      str+= 'Connects to: %s at coordinates (%.6f, %.6f, %.6f)\n'%(ct,x,y,z)
    match self.part_type:
      case 'CURVE':
        x,y=self.center
        str+='Center: (%.6f, %.6f)\n'% (x,y)
        str+='Length: %.6f\n'%(self.length)
        str+='Angular Length: %.6f\n'%(self.angular_length)
        str+='Revolutions: %.6f\n'%(self.revs)
      case 'STRAIGHT':
        str+='Length: %.6f\n'%(self.length)
      case 'TURNOUT':
        for path in self.paths:
          p=' '.join('%d'% a for a in path['steps'])
          str+= 'Path: %s: %s (length: %.6f)\n'%(path['text'],p,path['length'])
        for segment in self.segments:
          match segment['type']:
            case 'S':
              s=' '.join(fmt_point(segment[a]) for a in ['point1','point2'])
              str+= '%s points: %s\n'%(segment['type'],s)
            case 'C':
              a='center'
              c='%s: %s'%(a,fmt_point(segment[a]))
              s=['%s: %.6f'%(a,segment[a]) for a in ['radius','angle']]
              s.insert(1,c)
              str+= '%s points: %s\n'%(segment['type'],' '.join(s))
          
    return str
  def set_center(self,x,y):
    '''for curve set center'''
    self.center=[x,y  ]
  def add_end_point(self,connects_to,x,y,elev_defined): # connects_to is None for E4
    self.end_points+=[[connects_to,x,y,elev_defined]]
    match self.part_type:
      case 'STRAIGHT':
        if len(self.end_points)==2:
          self.length=get_length(self.end_points[0][1:3],self.end_points[1][1:3])
      case 'CURVE':
        if len(self.end_points)==2:
          self.angular_length=get_angle(a=self.end_points[0][1:3],b=self.center,c=self.end_points[1][1:3])
          radius=get_length(self.end_points[0][1:3],self.center)
          self.length=(self.angular_length/360)* 2* math.pi * radius
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

  def add_curve_segment(self,radius,center,angle):
    '''for turnouts only, segment type C or C3 ?
    center is an x,y pair'''
    assert self.part_type=='TURNOUT'
    segment={'type':'C','radius':radius,'center':center,'angle':angle}
    self.segments+=[segment]

  def add_path(self,label,steps):
    '''Add a path through the part
    steps is a list of segment numbers origin 1'''
    assert self.part_type=='TURNOUT'
    path={ 'text':label,'steps':steps,'length':0}
    self.paths+=[path]
  
  def  validate_turnout(self):
    '''once all the paths and segments are input, validate and cacluate length'''
    assert self.part_type=='TURNOUT'
    for path in self.paths:
      steps=path['steps'] 
      path_len=0
      for step in steps:
        segment=self.segments[step-1]
        match segment['type']:
          case 'S':
            path_len+=get_length(segment['point1'],segment['point2'])
          case 'C':
            path_len+=(segment['angle']/360)* 2* math.pi * segment['radius']
      path['length']=path_len

  def has_connection_to(self,id):
    return id in [a[0] for a in self.end_points]
  def mono_to_connection(self,conn_id):
    '''from the first connection to this one return true if height increases or stays the same'''
    h=[a[1]for a in self.end_points if a[0]==conn_id]
    return h[0] >= self.end_points[0][1]

def get_angle(a, b, c):
  '''Angular length for helix.  b is the center, a and c are the wings'''
  ang = math.degrees(math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0]))
  return abs(ang)

def read_input(in_file) -> list:
  '''parse the input file into a list of track parts'''
  recs=[]
  regex_quoted='"[^"]*"'
  pattern_quoted=re.compile(regex_quoted)

  with open(path.expanduser(in_file),encoding='windows-1252') as f:
    in_part=False
    for in_line in f.readlines():
      line=re.sub(r'  ',r' ',in_line)# there are some doubled spaces
      if in_part: # we are inside a part, process the endpoints
        assert line[0]=='\t'
        fields=line[1:].split(' ')
        match fields[0]:
          case  'T4': 
            connects_to=int(fields[1])
            elev_defined=float(fields[9])
            x,y=[float(fields[a]) for a in (2,3)] # the x & y coords
            part.add_end_point(connects_to=connects_to,x=x,y=y,elev_defined=elev_defined)
          case 'E4':#  E4 = unconnected)
            connects_to=None
            elev_defined=float(fields[8])
            x,y=[float(fields[a]) for a in (1,2)] # the x & y coords
            part.add_end_point(connects_to=connects_to,x=x,y=y,elev_defined=elev_defined)
          case 'P':
            part.add_path(fields[1],[float(a) for a in fields[2:]])
          case 'S':
            point_data=[float(a) for a in fields[2:]]
            part.add_straight_segment((point_data[0:2]),(point_data[2:4]))
          case 'C':
            radius,center_x,center_y,angle=[float(a) for a in fields[3:7]]          
            part.add_curve_segment(radius,(center_x,center_y),angle)
          case 'END$SEGS\n':
            recs+=[part]
            in_part=False
      else: # not in a part
        fields=line.strip().split(' ')
        if fields[0]in Part.part_types:
          part=Part(int(fields[1]),fields[0])
          for match in pattern_quoted.findall(line):
            part.turnout_detail=match[1:][:-1]
          if fields[0]=='CURVE':
            part.set_center(*[float(fields[a]) for a in (8,9)])
          in_part=True



    return recs

def sort_connected(df): 
  '''put adjacent tracks in order'''
  new_df=pd.DataFrame(columns=df.columns)
  usage=new_df[['part_index','connects_to']].value_counts()
  parts_to_do=[df.part_index.min()]
  while len(parts_to_do)>0:
    current_part=parts_to_do.pop(0)
    logging.debug(f'current part: {current_part}')
    end_points=df.loc[df.part_index==current_part]
    logging.debug (f'   endpoints:\n{end_points}')
    for idx,row in end_points.iterrows(): # loop through endpoints for this part and put at end of queue
      conn_to_part=row['connects_to']
      enqueue=conn_to_part not in new_df.part_index.to_list() # potentially add to queue
      exists=((new_df.part_index==row.part_index) & (new_df.connects_to==row.connects_to))
      if exists.sum()==0:
        logging.debug(f'adding ({row.part_index},{row.connects_to})')
        new_df=pd.concat([new_df,end_points.loc[[idx]]]) 
        usage=new_df[['part_index','connects_to']].value_counts()

      if (conn_to_part,current_part) in usage.index: # if the first direction is already used, skip
        continue # if both ends are accounted for skip this row   
      if enqueue:
        parts_to_do.insert(0,conn_to_part) # insert in queue
    logging.debug(f'queue is now {parts_to_do}')
  new_df.reset_index(drop=True,inplace=True)
  return new_df


def decorate(df) -> pd.DataFrame:
  '''Put the T for track on the part numbers'''
  for fld in ['part_index','connects_to']:
    df[fld]='T' + df[fld].astype(str)
  return df

def write_csv(df,out_file):
  '''write out the csv'''
  df.to_csv(out_file,sep=',')
  logging.debug (f'Wrote output to: {out_file}')

def main():
  '''the main routine'''


  with open ('config.yaml',encoding='UTF-8') as f:
    config=yaml.safe_load(f)

  in_file=config['pwd']+path.sep+config['xtc']
  out_file=config['pwd']+path.sep+config['csv']
  df=read_input(in_file=in_file)
  logging.info (f'input file has {df.shape[0]} rows')
  df=sort_connected(df)
  #df=monotonic_groups(df)
  df=decorate(df)
  logging.info (f'sorted file has {df.shape[0]} rows')
  write_csv(df,out_file)

if __name__=='__main__':
  main()
