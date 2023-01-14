#!/usr/bin/env python3
'''Read connected track defined and computed elevations out of the xtrkcad file and save into a csv'''
import logging
from os import path
import re
import yaml
import pandas as pd

logging.basicConfig(level=logging.INFO)

def get_df(in_file):
  '''parse the input file into a dataframe'''
  recs=[]

  with open(path.expanduser(in_file),encoding='windows-1252') as f:
    flag=False
    part_type=idx=None
    turnout_detail=None
    connects_to=None
    for in_line in f.readlines():
      regex=r'  '
      line=re.sub(regex,r' ',in_line)
      if flag:
        assert line[0]=='\t'
        fields=line[1:].split(' ')
        if fields[0]=='T4':
          connects_to=int(fields[1])
          elevs=[round(float(fields[a]),3) for a in (9,13)]
          row=[idx,connects_to,part_type]+elevs+[turnout_detail]
          recs+=[row]
        if fields[0]=='END$SEGS\n':
          flag=False
      else:
        fields=line.strip().split(' ')
        if fields[0]in ['STRAIGHT','CURVE','TURNOUT']:
          part_type=fields[0]
          idx=int(fields[1])
          regex='"[^"]*"'
          pattern=re.compile(regex)
          turnout_detail=None
          for match in pattern.findall(line):
            turnout_detail=match[1:][:-1]
            
          flag=True

  df =pd.DataFrame(recs)
  df.columns=['part_index','connects_to','part_type','elev_defined','elev_computed','turnout_detail']
  df.index.name='serial'
  return df

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
  df=get_df(in_file=in_file)
  logging.info (f'input file has {df.shape[0]} rows')
  df=sort_connected(df)
  logging.info (f'sorted file has {df.shape[0]} rows')
  write_csv(df,out_file)

if __name__=='__main__':
  main()
