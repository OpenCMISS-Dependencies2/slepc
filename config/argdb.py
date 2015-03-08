#
#  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#  SLEPc - Scalable Library for Eigenvalue Problem Computations
#  Copyright (c) 2002-2014, Universitat Politecnica de Valencia, Spain
#
#  This file is part of SLEPc.
#
#  SLEPc is free software: you can redistribute it and/or modify it under  the
#  terms of version 3 of the GNU Lesser General Public License as published by
#  the Free Software Foundation.
#
#  SLEPc  is  distributed in the hope that it will be useful, but WITHOUT  ANY
#  WARRANTY;  without even the implied warranty of MERCHANTABILITY or  FITNESS
#  FOR  A  PARTICULAR PURPOSE. See the GNU Lesser General Public  License  for
#  more details.
#
#  You  should have received a copy of the GNU Lesser General  Public  License
#  along with SLEPc. If not, see <http://www.gnu.org/licenses/>.
#  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#

import sys

class ArgDB:

  def __init__(self,argv):
    # standardize options
    for l in range(1,len(argv)):
      name = argv[l]
      if name.startswith('--enable'):
        argv[l] = name.replace('--enable','--with')
        if name.find('=') == -1: argv[l] += '=1'
      elif name.startswith('--disable'):
        argv[l] = name.replace('--disable','--with')
        if name.find('=') == -1: argv[l] += '=0'
        elif name.endswith('=1'): argv[l].replace('=1','=0')
      elif name.startswith('--without'):
        argv[l] = name.replace('--without','--with')
        if name.find('=') == -1: argv[l] += '=0'
        elif name.endswith('=1'): argv[l].replace('=1','=0')
      elif name.startswith('--with'):
        if name.find('=') == -1: argv[l] += '=1'
    self.argdb = argv[1:]

  def PopString(self,keyword):
    string = ''
    numhits = 0
    while True:
      found = 0
      for i, s in enumerate(self.argdb):
        if s.startswith('--'+keyword+'='):
          string = s.split('=')[1]
          found = 1
          numhits = numhits + 1
          del self.argdb[i]
          break
      if not found:
        break
    return string,numhits

  def PopPath(self,keyword):
    string = ''
    numhits = 0
    while True:
      found = 0
      for i, s in enumerate(self.argdb):
        if s.startswith('--'+keyword+'='):
          string = s.split('=')[1].rstrip('/')
          found = 1
          numhits = numhits + 1
          del self.argdb[i]
          break
      if not found:
        break
    return string,numhits

  def PopUrl(self,keyword):
    value = False
    string = ''
    numhits = 0
    while True:
      found = 0
      for i, s in enumerate(self.argdb):
        if s.startswith('--'+keyword):
          value = not s.endswith('=0')
          try: string = s.split('=')[1]
          except IndexError: pass
          found = 1
          numhits = numhits + 1
          del self.argdb[i]
          break
      if not found:
        break
    return string,value,numhits

  def PopBool(self,keyword):
    value = False
    numhits = 0
    while True:
      found = 0
      for i, s in enumerate(self.argdb):
        if s.startswith('--'+keyword+'='):
          value = not s.endswith('=0')
          found = 1
          numhits = numhits + 1
          del self.argdb[i]
          break
      if not found:
        break
    return value

  def PopHelp(self):
    value = False
    numhits = 0
    while True:
      found = 0
      for i, s in enumerate(self.argdb):
        if s.startswith('--h') or s.startswith('-h') or s.startswith('-?'):
          value = True
          found = 1
          numhits = numhits + 1
          del self.argdb[i]
          break
      if not found:
        break
    return value

  def ErrorIfNotEmpty(self):
    if self.argdb:
      sys.exit('ERROR: Invalid arguments '+' '.join(self.argdb)+'\nUse -h for help')
    
