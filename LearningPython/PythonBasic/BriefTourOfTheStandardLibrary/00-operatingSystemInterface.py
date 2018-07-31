'''
2018年7月17日17:35:54
author：sierkinhane
'''

import os
import shutil
import glob
import sys

cwd = os.getcwd() # Return the current working directory
# print('{0}'.format(cwd))
#os.chdir('../') # Change current working directory
#os.system('mkdir today') # Run the command mkdir in the system shell
# print(dir(os)) # returns a list of all module functions
# print(help(os)) # returns an extensive manual page created from the module's docstrings

#shutil.copyfile('data.db', 'archive.db')
#shutil.move('./1', '../') #move the file

# File Wildcards

py_file = glob.glob('*.py') # reutrn a list
#print('all py_file are:{}'.format(py_file))

# Command Line Arguments
#print(sys.argv) # get the command arguments

# 系统编译路径
os_path = os.path
print(os_path)