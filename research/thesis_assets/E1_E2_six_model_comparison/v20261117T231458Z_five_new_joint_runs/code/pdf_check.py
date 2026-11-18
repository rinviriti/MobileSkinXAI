"""Validate classic Matplotlib PDF cross-references without installing a PDF viewer."""
from pathlib import Path
import re

def verify_pdf(path):
 data=Path(path).read_bytes();assert data.startswith(b'%PDF-') and data.rstrip().endswith(b'%%EOF'),path
 match=re.search(rb'startxref\s+(\d+)\s+%%EOF\s*$',data);assert match,path;offset=int(match.group(1));assert data[offset:offset+4]==b'xref',path
 lines=data[offset:].splitlines();i=1;objects=0
 while not lines[i].startswith(b'trailer'):
  first,count=map(int,lines[i].split());i+=1
  for j in range(count):
   pos,gen,kind=lines[i].split();i+=1
   if kind==b'n':
    start=int(pos);expected=f'{first+j} {int(gen)} obj'.encode();assert data[start:start+len(expected)]==expected,(path,first+j);objects+=1
 assert b'/Root ' in b'\n'.join(lines[i:]);assert re.search(rb'/Type\s*/Page\b',data),path
 return objects
