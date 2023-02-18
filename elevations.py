#!/usr/bin/env python3
'''Read connected track defined and computed elevations out of the xtrkcad file and save into a csv'''
import logging
import math
from os import path
import re

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import yaml


logging.basicConfig(level=logging.INFO)

#files
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
            x,y,angle=[float(fields[a]) for a in (2,3,4)] # the x & y coords
            part.add_end_point(connects_to=connects_to,x=x,y=y,elev_defined=elev_defined,angle=angle)
          case 'E4':#  E4 = unconnected)
            connects_to=None
            elev_defined=float(fields[8])
            x,y,angle=[float(fields[a]) for a in (1,2,3)] # the x & y coords, and the angle
            part.add_end_point(connects_to=connects_to,x=x,y=y,elev_defined=elev_defined,angle=angle)
          case 'P':
            part.add_path(fields[1],[int(a) for a in fields[2:]])
          case 'S':
            point_data=[float(a) for a in fields[3:]]
            part.add_straight_segment((point_data[0:2]),(point_data[2:4]))
          case 'C':
            radius,center_x,center_y,angle,swing=[float(a) for a in fields[3:8]]          
            part.add_curve_segment(radius,(center_x,center_y),angle,swing)
          case 'END$SEGS\n':
            recs+=[part]
            in_part=False
      else: # not in a part
        fields=line.strip().split(' ')
        if fields[0]in Part.part_types:
          part=Part(int(fields[1]),fields[0])
          for match in pattern_quoted.findall(line):
            part.mfg_info=match[1:][:-1].split('\t')
          if fields[0]=='CURVE':
            part.set_center(*[float(fields[a]) for a in (8,9)])
          if fields[0]=='TURNOUT':
            # get the base location (aka 'orig') and angle
            part.turnout_orig=[float(f) for f in fields[8:11]]# x,y,z
            part.turnout_angle=float(fields[11])
          in_part=True

    return recs
def write_csv(df,out_file):
  '''write out the csv'''
  df.to_csv(out_file,sep=',')
  logging.debug (f'Wrote output to: {out_file}')

# formatters
def decorate(df) -> pd.DataFrame:
  '''Put the T for track on the part numbers'''
  for fld in ['part_index','connects_to']:
    df[fld]='T' + df[fld].astype(str)
  return df
def fmt_node_id(a,b):
  '''create string node-id 
  argument: two integers, which are the part IDs that are connected
    If less than zero it indicates that it is not connected. 
  returns: a string with the lower valid part number 1st followed by a hyphen and the higher part number,
    unless there is a negative, in which case the unconnected item comes 2nd
    Decorators T (connected) and E (ends) are used to make them node name more readable.
  examples:
    T86-T97 - The place where parts 86 and 97 connect.
    T101-E5 - A siding that ends
    E9-E6 - a loose piece of track
    T86-T97a - TODO distinguish when pathologically two tracks connect to each other
  '''
  ab=np.asarray([a,b])
  match sum(np.sign(ab)):
    case 2:
      r='T%d-T%d'%(min(ab),max(ab))
    case 0:
      r='T%d-E%d'%(ab.max(),abs(ab.min()))
    case -2:
      r='E%d-E%d'%(abs(ab.min()),abs(ab.max()))
  return r 
def fmt_point(point)->str:
  '''format pairs as points'''
  x,y=point
  return '(%.6f, %.6f)'%(x,y)

# geometry functions
def get_angle(a, b, c):
  '''Angular length for helix.  b is the center, a and c are the wings'''
  ang = math.degrees(math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0]))
  return abs(ang)
def get_length(point1,point2):
  '''distance between two points'''
  point1=np.array(point1)
  point2=np.array(point2)
  distance=(np.subtract(point1,point2)**2).sum()**.5
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
    self.segments=[] # in the form ((x1,y1),(x2,y2))
    # only for turnouts where there may will be more than 1
    self.paths=[] # a list of dicts with keys: "text","steps","length"
    # stepselements are indices into the segments origin 1 (as they are in xtrkcad)
    # for left empty for straights and curves
    self.group=0
    self.length=0
    self.center=None
    self.angular_length=None
    self.mfg_info=None
    self.turnout_orig=None
    self.turnout_angle=None
    self.revs=None # revolutions for helix
  
  def __str__(self):
    '''return code + part number where code S=straight, C=curve, T=Turnout'''
    return '%s %d'%(self.part_type[0:1],self.id)  
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




def step_len_div(part) -> list:
  '''for a turnout part, inspect the paths, get the length and the angle of divergence for each path
  returns list lengths, and list of divergenced in degrees
  '''
  lens=[]
  divs=[]
  for path in part.paths:
    steps=path['steps']  
    path_len=[]
    step_divergence=[]
    for sn in steps:
      seg=part.segments[sn-1]
      match seg['type']: # get the length for all segments along path
        case 'S':
          step_end_point=seg['point2']
          path_len+=[abs(get_length(seg['point1'],step_end_point))]
          step_divergence+=[0]
        case 'C': # in the case of curves also alter the angle
          path_len+=[(seg['swing']/360)* 2* math.pi * abs(seg['radius'])]
          step_divergence+=[seg['swing']*np.sign(seg['radius'])]
    lens+=[sum(path_len)]
    divs+=[sum(step_divergence)]
  return lens,divs

def parts_to_graph(parts)-> nx.Graph:
  '''convert to graph so we can find paths between defined elevations and thus compute slopes and elevations where they are not defined'''
  G=nx.Graph()
  unconnected=0 # replace unconnected endpoint reference numbers with negatives
  for part in parts:
    match part.part_type:
      case 'STRAIGHT' | 'CURVE':
        end_points=[a[0] for a in part.end_points]
        node_ids=[]
        for end_point in end_points:
          if end_point is None:
            unconnected+=-1
            end_point=unconnected
          node_ids+=[fmt_node_id(part.id,end_point)]
        u,v=node_ids
        G.add_edge(u,v,length=part.length,part=str(part))
      case 'TURNOUT':
        # match up the connections to the paths by
        # 1) finding the set of segments that are shared by paths (assert this is singular)
        # 2) locate this in the connections pool by start point
        # 3) Compute the divergence in degrees for each path
        # 4) Use the divergence angle as the index into the connection pool to complete the mapping
        common_segments=list(set.intersection(*[set(p['steps']) for p in part.paths]))
        assert len(common_segments)==1
        start_point=np.asarray(part.turnout_orig[0:2]) # this is the actual start point (no offset)

        # 2) locate this in the connections pool by start point
        ep_df=pd.DataFrame(part.end_points)
        ep_df.columns=['connects_to','x','y','z','angle']
        sel=(ep_df[['x','y']]==start_point).all(axis=1)
        assert 1==sel.sum()
        node_start,angle_start=ep_df.loc[sel,['connects_to','angle']].squeeze().tolist()
        from_node_id=fmt_node_id(part.id,node_start)
        path_lengths,path_divergences=step_len_div(part)

        pdl=pd.DataFrame({'divergence':path_divergences,'length':path_lengths})
        for _,row in pdl.iterrows():
          angle_end=row['divergence']+(180+angle_start)%360 # the other end is going the opposite direction
          sel=np.isclose(ep_df['angle'],angle_end,.001)
          assert 1==sum(sel)
          to=sorted([part.id,ep_df.loc[sel,'connects_to'].squeeze()])
          to_node_id=fmt_node_id(*to)
          G.add_edge(from_node_id,to_node_id,length=row['length'],part=str(part))
  logging.info('Converted to graph')
  return G      

def draw_graph(G,path):
  '''Draw the graph and store at path'''
  plt.figure(figsize=(20,10),)
  plt.axis('off')
  plt.tight_layout(pad=0)
  edge_labels = dict([((n1, n2), d['part']) for n1, n2, d in G.edges(data=True)])
  #pos=nx.spring_layout(G,k=0.16)
  pos=nx.nx_pydot.graphviz_layout(G,prog='neato')
  nx.draw_networkx(G, pos,with_labels=False,node_size=50)
  nx.draw_networkx_edge_labels(G,pos,edge_labels=edge_labels,font_size=10)
  plt.savefig(path)
  logging.info('Displayed graph in file %s'%(path))

def main():
  '''the main routine'''
  with open ('config.yaml',encoding='UTF-8') as f:
    config=yaml.safe_load(f)
  in_file=config['pwd']+path.sep+config['xtc']
  graph_file=config['docs']+path.sep+config['graph_file']
  parts=read_input(in_file=in_file)
  logging.info (f'input file has {len(parts)} parts')
  G=parts_to_graph(parts)
  draw_graph(G,graph_file)


if __name__=='__main__':
  main()
