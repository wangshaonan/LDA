#!/usr/bin/python
#coding = utf-8
import os
import os.path
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
from lxml import etree

corpuslist = os.walk(sys.argv[1])
#print corpuslist
corpuswrite = open('renminrb', 'a')
for root, dirs, corpusfiles in corpuslist:
    for name in corpusfiles:
        files = os.path.join(root, name)
        print files
        html = open(files, 'r').read()
        try:
            page = etree.HTML(html.lower().decode('UTF-8', 'ignore'))
#            page = etree.HTML(html.lower()) 
        except UnicodeDecodeError as err:
            continue
            
        if page:
            text = page.xpath(u'//p')
            for t in text:
#                print t.text
                corpuswrite.write(str(t.text))
                corpuswrite.write('\n')
