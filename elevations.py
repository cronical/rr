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
  for in_line in f.readlines():
    line=re.sub(regex,r' ',in_line)
    if flag:
      assert line[0]=='\t'
      fields=line[1:].split(' ')
      if fields[0]=='T4':
        elevs=[float(fields[a]) for a in (9,13)]
      recs[idx]=[track_type]+elevs
      if fields[0]=='END$SEGS\n':
        flag=False
    else:
      fields=line.strip().split(' ')
      if fields[0]in ['STRAIGHT','CURVE','TURNOUT']:
        track_type=fields[0]
        idx=int(fields[1])
        flag=True

df =pd.DataFrame(recs).T
df.columns=['track','defined','computed']
print(df)
