import os

cmd = "evo_ape kitti gt.kitti noise.kitti  -va --plot --plot_mode xy"

os.system(cmd)

cmd2 = "evo_ape kitti gt.kitti opt.kitti  -va --plot --plot_mode xy"
os.system(cmd2)