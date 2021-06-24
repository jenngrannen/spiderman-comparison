#!/usr/bin/env python
import os

if __name__=="__main__":
    cmd = "nvidia-docker run -it -v %s:/host \
                                 -v %s:/host/data jenn-density" % (os.path.join(os.getcwd(), '..'), '/raid/jennifer/density-classification/')
    #cmd = "nvidia-docker run -it jenn-density" 
    #cmd = "docker run --runtime=nvidia -it -v %s:/host jenn-density" % (os.path.join(os.getcwd(), '..'))
    code = os.system(cmd)
