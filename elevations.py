#!/usr/bin/env python3
'''Read connected track defined and computed elevations out of the xtrkcad file and save into a csv'''
import logging
from math import isnan
from os import path

import re

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import networkx as nx
import numpy as np
import pandas as pd
import yaml

from part_class import Part,fmt_node_id

color_na='tab:red'
COLOR_SET=list(set(mcolors.TABLEAU_COLORS.keys())-set([color_na]))

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

def parts_as_dict(parts_list):
  '''convert parts list to a dict'''
  parts={}
  for part in parts_list:
    parts[part.id]=part
  return parts

def parts_to_graph(parts):
  '''convert parts dict to graph so we can find paths between defined elevations and thus compute slopes and elevations where they are not defined
  while we are at it, also accumulate the positions of the nodes.
  returns the graph and a dictionary of physical node positions
  '''
  G=nx.Graph()
  pos={}
  unconnected=0 # replace unconnected endpoint references with sequential numbers
  for _,part in parts.items():
    match part.part_type:
      case 'STRAIGHT' | 'CURVE':
        connects_to=[a[0] for a in part.end_points]
        nodes={}# node_id & heights
        for ix,adjacent_part_id in enumerate(connects_to):
          if adjacent_part_id is None:
            unconnected+=1
            node_id=fmt_node_id(part,end_ref=unconnected)
          else:
            node_id=fmt_node_id(part,connects_to=parts[adjacent_part_id])  
          nodes[node_id]=part.end_points[ix][3]# the height or zero if not yet known
          pos[node_id]=np.array(part.end_points[ix][1:3]) # the x and y values. Gets both ends due to loop
          # if its connected, this will happen twice
        for node_id,height in nodes.items():
          G.add_node(node_id,height=height)
        u,v=nodes.keys()
        G.add_edge(u,v,length=part.length,part_id=str(part),weight=part.length)
        pass
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
        from_node_id=fmt_node_id(parts[part.id],connects_to=parts[node_start])
        xyz=np.array(ep_df.loc[sel,['x','y','z']].squeeze())
        from_height=xyz[2]
        pos[from_node_id]=xyz[0:2]
        path_lengths,path_divergences=step_len_div(part)
        G.add_node(from_node_id,height=from_height)

        pdl=pd.DataFrame({'divergence':path_divergences,'length':path_lengths})
        for ix,row in pdl.iterrows():
          angle_end=row['divergence']+(180+angle_start)%360 # the other end is going the opposite direction
          sel=np.isclose(ep_df['angle'],angle_end,.001)
          assert 1==sum(sel)
          to=ep_df.loc[sel,'connects_to'].squeeze()
          to_node_id=fmt_node_id(part,parts[to])
          xyz=np.array(ep_df.loc[sel,['x','y','z']].squeeze())
          pos[to_node_id]=xyz[0:2]
          part_path=str(part)+'-%d'%(ix+1)
          G.add_node(to_node_id,height=xyz[2])
          G.add_edge(from_node_id,to_node_id,length=row['length'],part_id=part_path,weight=1/row['length'])
        pass
  for node in G.nodes:
    assert node in pos,'missing position'  
  logging.info('Converted to graph')
  return G,pos      

def walk_tree(tree,current_node,processed=[],node_line=[],depth=0,next_ramp=0):
  '''follow each branch/stem on the tree
  tree is a spanning tree as a networkx graph
  current_node is the node name (start with the root)
  processed is a list of nodes already processed (used to filter to prevent loops)
  node_line is an array of node ids and height pairs that accumulate until a slope can be calculated
  depth tracks recursion
  next_ramp is the number to associate with the next ramp 
  returns 
    processed, 
    node_line as revised, 
    epe = endpoints encountered
    next_ramp
  '''
  default_height=0.5
  done=False
  logging.debug('entering walk at stack depth = %d'% depth)
  processed.append(current_node)
  epe=0
  if depth==0:
    node_info=dict(tree[current_node])
    node_info.setdefault('height',0)
    node_line.append([current_node,node_info['height']])
    logging.info('Start node: %s. height=%.3f'%(current_node,node_line[-1][1]))
  while not done:
    paths=dict(tree[current_node])
    for node_id in set(paths.keys()).intersection(set(processed)): # remove the path we came in on
      del paths[node_id]
    if len(paths)==0: # if there are no paths exit this recursion level
      done=True
      continue

    # otherwise, take each path in turn
    for node_id,path_attr in paths.items():
      node_info=dict(tree.nodes[node_id])
      height=node_info['height']
      logging.debug('Node: %s. height=%.3f'%(node_id,height))
      logging.debug('      Edge %s: length = %.3f'%(path_attr['part_id'],path_attr['length']))

      # add the node on this path to the stack
      node_line.append([node_id,height])
      df=pd.DataFrame(node_line,columns=['node1','height'])
      df.set_index('node1',inplace=True)
      assert df.height.isnull().values.any()==False, 'Bad height'# pylint: disable=singleton-comparison

      # determine if this is an end point in the spanning tree
      neighbor_count=len(list(tree.neighbors(node_id)))
      if neighbor_count==1: # this node has no neighbors other than the path we came in on
        if height==0: # default the height of this node if neede be
          sel=df.height!=0
          if not sel.values.any():
            logging.warning('no heights in node line... using default')
            df['height']=default_height                            
            sel=df.height!=0
          last_height=df.loc[sel].tail(1)['height'].squeeze()
          height=last_height
          df.iloc[-1,df.columns.get_loc('height')]=height

      # when we have hit a node with a defined height, interpolate
      if height !=0: 
        #Node %s has non-zero height.

        # if this is the only defined height, default start to make it level with this node's value
        if (df.height==0).head(-1).values.all():  
          node_line[0][1]=height
          df.iloc[0,df.columns.get_loc('height')]=height
        

        # number of nodes to and including the most recent one with a height
        ramp_size=2+list(reversed((df.height.head(-1)>0).to_list())).index(True) # 2 = 1 for origin and 1 for this node
        df=df.tail(ramp_size) # this is the section that has the same slope
        start_height=df.head(1).height.squeeze()
        nodes=df.index.to_list()
        edge_ids=list(zip(nodes[:-1],nodes[1:])) # all the edges in the same slope section
        edge_lens=[0]*len(edge_ids) # no great way found to get the length values for the edges without looping
        for e, datadict in tree.edges.items():
          for edg in (e,(e[1],e[0])):
            if edg in edge_ids:
              edge_lens[edge_ids.index(edg)]=datadict['length']
        df['length']=0
        df.loc[nodes[:-1],'length']=edge_lens # all the found lengths are now in the dataframe

        if (df.length.head(-1)==0).values.any():  # display those not found
          logging.error('Bad length at')
          logging.error(df.loc[df.length==0])
        start_height=df.head(1).height
        ramp_len=df.length.sum().squeeze()
        slope=(height-start_height).squeeze()/ramp_len
        s,e=df.index.values[1],df.index.values[-1]
        logging.info('Ramp %d. %d edges: [%s...%s] Heights: %.3f %.3f. Tot len: %.3f. Grade: %.3f'%(next_ramp,ramp_size-1,s,e,start_height,height,ramp_len,slope*100))

        # die if the total length of the section is zero.
        assert not isnan(slope), 'Bad slope'# pylint: disable=singleton-comparison

        # track if height was computed to allow coloring. 1 is this was computed, 0 for defined
        df['computed']=(df.height==0).astype(int) 

        # put the computed heights in the dataframe
        df['cum']=df.length.cumsum(axis=0).shift(fill_value=0)
        df['height']=df.cum.apply(lambda x: start_height+x*slope)

        # attributes into a dict
        att=df[['height','computed']].to_dict('index')
        # and then into the spanning tree
        nx.set_node_attributes(tree,att)
        logging.debug('Set height for %d nodes'%df.shape[0])

        # mark the edges with the ramp (section) number and the grade
        df=pd.DataFrame(columns=['edge_id'])
        df['edge_id']=edge_ids
        df.set_index('edge_id',inplace=True)
        df['ramp_id']=next_ramp
        df['grade']=100*slope
        att=df[['ramp_id','grade']].to_dict('index')
        nx.set_edge_attributes(tree,att)
        logging.debug('Set ramp_id & grade for %d edges'%df.shape[0])
        next_ramp+=1

      # if there is a forward path take it
      if neighbor_count>1:
        processed,node_line,epf,next_ramp=walk_tree(tree,node_id,processed,node_line,depth=depth+1,next_ramp=next_ramp)
        epe+=epf
      else:
        epe+=1
      removed=node_line.pop()
      logging.debug('Removed line node %s'%removed[0])
    done=True
  logging.debug('exiting walk at stack depth = %d'% depth)
  if depth==0:
    removed=node_line.pop()
    logging.debug('Removed line node %s'%removed[0])
  return processed,node_line,epe,next_ramp

def walk_dfs_list(G,dfs_list):
  '''walk the results of edge_dfs to to identify ramps, compute new node heights and edge grades
  '''
  next_ramp=0
  default_height=0.5
  df=pd.DataFrame(columns=['node1','node2','height','length','computed','cum','ramp_id','grade'])
  for seq,edge_id in enumerate(dfs_list):
    # add a row into the working dataframe for this edge
    u,v=edge_id  
    height=G.nodes[u]['height']
    length=G.edges[edge_id]['length']
    df.loc[len(df.index),['node1','node2','height','length']]=[u,v,height,length]

    #When  the section ends, the next edge is not adjacent to node2
    #this happens when the depth first scan reaches and end and goes back to a previous branch.
    next_edge=(dfs_list+['PAST END OF LIST'])[seq+1]
    adj=list(G.neighbors(v)) # adjacent nodes of the to node
    if next_edge[1] in adj:
      logging.info('%d. Continuing from %s'%(seq,v))
    else:
      logging.info('Section ended on %s'%v)
      logging.info('Section size is %d'%df.shape[0])

      # add a row for the last node
      df.loc[len(df.index),['node1','height']]=[v,G.nodes[v]['height']]

      # track if height was computed to allow coloring. 1 is this was computed, 0 for defined
      df['computed']=(df.height==0).astype(int) 

      # within the section make sure the first and last items are non-zero
      # if 1st height is zero, fill with first non zero or default
      if df.head(1)['height'].squeeze()==0:
        df.at[df.index[0],'height']=(df.loc[df.height!=0,'height'].to_list()+[default_height])[0]
        n,h=df.head(1)[['node1','height']].values.flatten()
        logging.warning('Node %s had zero height. Using %.3f'%(n,h))

      # if the last one is zero, fill with most recent non zero height  
      if df.tail(1)['height'].squeeze()==0:
        df.at[df.index[-1],'height']=df.loc[df.height!=0,'height'].to_list()[-1]
        n,h=df.tail(1)[['node1','height']].values.flatten()
        logging.warning('Node %s had zero height. Using %.3f'%(n,h))

      # break section into ramps
      df['ramp_id']=(next_ramp-1)+(df.height!=0).astype(int).cumsum(axis=0)
      ramps=pd.unique(df.ramp_id.head(df.shape[0]-1))# last row is the trailing node, not an edge
      for ramp in ramps:
        sel=df.ramp_id==ramp
        ramp_len=df.loc[sel].length.sum().squeeze()
        last_height=df.at[df.loc[sel].index.tolist()[-1]+1,'height']#ending height is on next row
        heights=df.loc[sel].height.to_list()+[last_height]
        slope=(heights[-1]-heights[0])/ramp_len
        # put the computed heights in the dataframe for this ramp
        df.loc[sel,'cum']=df.loc[sel].length.cumsum(axis=0).shift(fill_value=0)
        df.loc[sel,'height']=df.loc[sel].cum.apply(lambda x: heights[0]+x*slope)
        df.loc[sel,'grade']=slope*100
        logging.debug(df)
      next_ramp=1+df.ramp_id.max()

      # attributes into a dict
      # the last node may already have been seen
      seen=(df.loc[df.index[:-1],'node1']==df.at[df.index[-1],'node1']).any()
      if seen: # if so drop it so node1 can become an index
        df=df.head(df.shape[0]-1)
      df.set_index('node1',inplace=True)
      att=df[['height','computed']].to_dict('index')
      # and then into the graph
      nx.set_node_attributes(G,att)
      logging.debug('Set height for %d nodes'%(len(att)))

      # mark the edges with the ramp (section) number and the grade
      df.reset_index(inplace=True)
      dfe=df.head(df.shape[0]-[1,0][seen])# just the edges
      dfe['edge']=''
      for ix,row in dfe.iterrows():
        dfe.at[ix,'edge']=(row['node1'],row['node2'])  
      dfe.set_index('edge',inplace=True)    
      att=dfe[['ramp_id','grade']].to_dict('index')
      nx.set_edge_attributes(G,att)
      logging.debug('Set ramp_id & grade for %d edges'%dfe.shape[0])

      logging.info('%d. Starting new section from %s'%(seq,u))
      df=df.head(0)
  logging.info('walk_dfs_list complete.')


def draw_graph(G,pos,path=None,fig_num=1,node_labels=None,node_colors=None,edge_labels=None,edge_color=None,title=None):
  '''Draw the graph and store at path or show if None
  pos is used for the node locations'''
  plt.figure(figsize=(20,10),)
  #plt.axis('off')
  if title:
    plt.title(title)

  options={'node_color':node_colors,'with_labels':False,'node_size':40,'edge_color':edge_color,'width':3}
  nx.draw_networkx(G, pos,**options)
  if node_labels:
    options={'horizontalalignment':'right','verticalalignment':'top','font_size':6,'font_color':'k','font_weight':'bold'}
    nx.draw_networkx_labels(G,pos,node_labels,**options)
  if edge_labels:
    options={'font_size':6,'rotate':True,'font_color':'b'}
    nx.draw_networkx_edge_labels(G,pos,edge_labels=edge_labels,**options)
  if path:
    plt.savefig(path)
    logging.info('Displayed graph in file %s'%(path))
  else:
    plt.show(block=False)

def drawing_decorations(G,node_info,edge_info):
  '''given a graph G and parameters, return items suitable for networkx drawing
    args:
      G - a networkx graph
      node_info - a dict with node parms
      edge_info - a dict with edge parms
      the parm dicts are:
        key     value
        label   attribute name to use or None
        fmt     a % format style including the %
        color_by attribute name or None  (if given must be integer)
        color_set the set of colors to cycle through
        color_na the color to use for not available - should not be in color_set

    return: node_labels, node_colors, edge_labels, edge_colors
    '''
  result=[]
  
  for ix,info in enumerate([node_info,edge_info]):
    get_key_funcs=G.nodes(),G.edges()
    get_attr_funcs=nx.get_node_attributes,nx.get_edge_attributes
    labels=None
    if info['label'] is not None:
      labels=get_attr_funcs[ix](G,info['label'])
      labels={key:info['fmt']%(value) for (key,value) in labels.items()}
    result.append(labels)

    colors=None
    if info['color_by']is not None: # handles missing values
      key_list=list(get_key_funcs[ix])# list all nodes or edges
      if isinstance(key_list[0],tuple):
        df=pd.DataFrame.from_records(key_list)
      else:
        df=pd.DataFrame(key_list)
      df.set_index(list(df.columns),inplace=True)
      df['attribute']=0
      attribute=get_attr_funcs[ix](G,info['color_by'])
      for key,value in attribute.items():
        df.loc[key,'attribute']=value
      color_set=info['color_set']
      colors=df.attribute.to_list()
      for i,val in enumerate(colors):
        if pd.isna(val):
          colors[i]=info['color_na']
        else:
          colors[i]=color_set[int(val) % len(color_set)]
    result.append(colors)
  result=tuple(result)
  return result


def file_path(method,chart,config):
  '''Produce a file name or None as needed from args'''
  if config['show_only']:
    return None
  else:
    base=config['docs']
    file_ctrl={k:base+v for (k,v) in config.items() if k.endswith('file')}
    return file_ctrl['_'.join([method,chart,'file'])]


def main():
  '''the main routine'''
  with open ('config.yaml',encoding='UTF-8') as f:
    config=yaml.safe_load(f)
  in_file=config['pwd']+path.sep+config['xtc']
    
  parts=read_input(in_file=in_file)
  parts=parts_as_dict(parts)
  logging.info (f'input file has {len(parts)} parts')
  G,physical_pos=parts_to_graph(parts)
  logging.info('Graph has %d nodes'%len(list(G.nodes())))
  logging.info('Graph has %d edges'%len(list(G.edges())))

  # find a root on a siding
  edges=list(G.edges)
  root_candidates=[a for (a,b) in edges if 'E' in a]+[b for (a,b) in edges if 'E' in b]
  root=root_candidates[0]

  for method in ('dfs','st'):

    if method=='dfs':
      # DEPTH FIRST SEARCH METHOD
      dfs=list(nx.edge_dfs(G,root_candidates[0]))

      # set the heights (modifies G)
      walk_dfs_list(G,dfs)

      bi_color=[COLOR_SET[8],COLOR_SET[1]]# orange for computed, yellow for defined
      node_info={'label':'height','fmt':'%.2f','color_by':'computed','color_set':bi_color,'color_na':color_na}
      edge_info={'label':'ramp_id','fmt':'%d','color_by':'ramp_id','color_set':COLOR_SET,'color_na':color_na}
      for chart in ('ramps','physical','logical'):
        file_name=file_path(method,chart,config)
        if chart == 'ramps':
          # chart with sequence number labels on edges
          title='Sequences for depth first search starting at '+root_candidates[0]
          edge_labels={key:'%d'%(sequence) for sequence,key in enumerate (dfs)}
          options={'edge_labels':edge_labels,'title':title}
          pos=physical_pos
        if chart in ['physical','logical']:
          title=chart.title()+' view - based on depth first search'
          edge_info['label']='ramp_id'
          edge_info['fmt']='%d'
          node_labels, node_colors, edge_labels, edge_colors=drawing_decorations(G,node_info,edge_info)
          options={'node_labels':node_labels,'node_colors':node_colors,'edge_labels':edge_labels,'edge_color':edge_colors,'title':title}
        if chart=='physical':
          pos=physical_pos
        if chart=='logical':
          pos=nx.nx_pydot.graphviz_layout(G,prog='neato')
          options['edge_labels']=None
        draw_graph(G,pos,file_name,**options)

    if method =='st':
      # SPANNING TREE METHOD - TO SHOW ITS DEFECTS
      G,physical_pos=parts_to_graph(parts) # fresh copy of G

      # create a spanning tree so it can be walked to set heights
      st_ramps = nx.minimum_spanning_tree(G,weight='length')
      logging.info('Spanning tree node count= %d'%(len(list(st_ramps.nodes()))))

      # set the heights
      processed,node_line,epe,next_ramp=walk_tree(st_ramps,root)

      logging.info('Node line has %d members'%len(node_line))
      logging.info('Processed %d nodes'%len(processed))
      logging.info('Encountered %d endpoints'%epe)
      logging.info('Ramp numbers: 0 - %d'%(next_ramp-1))
      
      # transfer the attributes from the spanning tree to the main graph
      att=dict(st_ramps.nodes(data=True))
      nx.set_node_attributes(G,att)

      ea=list(st_ramps.edges(data=True))
      att={(a[0],a[1]): a[2]for a in ea}
      nx.set_edge_attributes(G,att)

      for chart in ('ramps','physical','logical'):
        file_name=file_path(method,chart,config)
        if chart == 'ramps':
          graph=st_ramps
          title='Spanning tree with ramps by color'
          pos=nx.nx_pydot.graphviz_layout(st_ramps,prog='neato')
          node_labels, node_colors, edge_labels, edge_colors=drawing_decorations(st_ramps,node_info,edge_info)
          options={'node_labels':node_labels,'node_colors':node_colors,'edge_labels':edge_labels,'edge_color':edge_colors,'title':title}

        if chart in ['physical','logical']:
          graph=G
          title=chart.title()+' view of the graph itself - based on spanning tree'
          node_labels, node_colors, edge_labels, edge_colors=drawing_decorations(G,node_info,edge_info)
          options={'node_labels':node_labels,'node_colors':node_colors,'edge_labels':edge_labels,'edge_color':edge_colors,'title':title}

        if chart=='physical':          
          pos=physical_pos

        if chart=='logical':
          pos=nx.nx_pydot.graphviz_layout(G,prog='neato')
          options['edge_labels']=None

        draw_graph(graph,pos,file_name,**options)
  input('Press enter to close: ')  
if __name__=='__main__':
  main()
