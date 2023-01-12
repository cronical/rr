#!/usr/bin/env python3
'''Read connected track defined and computed elevations out of the xtrkcad file and save into a csv'''
from os import path
import re
import yaml
import pandas as pd

with open ('config.yaml',encoding='UTF-8') as f:
  config=yaml.safe_load(f)

in_file=config['pwd']+path.sep+config['xtc']
recs={}

regex=r'  '

with open(path.expanduser(in_file),encoding='windows-1252') as f:
  flag=False
  track_type=idx=None
  sub=0
  for in_line in f.readlines():
    line=re.sub(regex,r' ',in_line)
    if flag:
      assert line[0]=='\t'
      fields=line[1:].split(' ')
      if fields[0]=='T4':
        elevs=[round(float(fields[a]),3) for a in (9,13)]
        recs[idx+(.1*sub)]=[track_type]+elevs
        sub=sub+1
      if fields[0]=='END$SEGS\n':
        flag=False
        sub=0
    else:
      fields=line.strip().split(' ')
      if fields[0]in ['STRAIGHT','CURVE','TURNOUT']:
        track_type=fields[0]
        idx=float(fields[1])
        sub=0 # sub index
        flag=True

df =pd.DataFrame(recs).T
df.columns=['track','defined','computed']
print(df)
