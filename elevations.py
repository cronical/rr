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

from part_class import Part,get_length,fmt_node_id

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
            if part.part_type=='TURNOUT':
              part.validate_turnout()
              pass
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

def step_len_div(part) -> list:
  '''for a turnout part, inspect the paths, get the length and the angle of divergence for each path
  returns list lengths, and list of divergenced in degrees
  '''
  lens=[]
  divs=[]
  for path in part.paths:
    lens+=[path['length']]
    divs+=[path['divergence']]
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
        G.add_edge(u,v,length=part.length,part_id=str(part))
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
          G.add_edge(from_node_id,to_node_id,length=row['length'],part_id=str(part))
  logging.info('Converted to graph')
  return G      

def draw_graph(G,path):
  '''Draw the graph and store at path'''
  plt.figure(figsize=(20,10),)
  plt.axis('off')
  plt.tight_layout(pad=0)
  edge_labels = dict([((n1, n2), d['part_id']) for n1, n2, d in G.edges(data=True)])
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
