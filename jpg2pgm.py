# coding=utf-8 
from PIL import Image
import os.path
import glob
 
num = 0
def jpg2pgm( jpg_file , pgm_dir , num ):
    # 首先打开jpg文件
    jpg = Image.open( jpg_file )
    jpg = jpg.convert('L')
    size = jpg.size
    sy = size[1] # height
    sx = size[0] # width
    # resize , 双线性插值
    jpg = jpg.resize( (sx,sy) , Image.BILINEAR )
    # 调用 python 函数 os.path.join , os.path.splitext , os.path.basename ，产生目标pgm文件名
    name =(str)(os.path.join( pgm_dir , os.path.splitext( os.path.basename(str(num)) )[0] ))+".pgm"
    # name =(str)(num)+".pgm"
    # 创建目标pgm 文件
    jpg.save( name )

 
# 将所有的jpg文件放在当前工作目录，或者 cd {存放jpg文件的目录}
for jpg_file in glob.glob('Random_Images_Dataset_1/images/*.jpg'):
    jpg2pgm( jpg_file , 'Random_Images_Dataset_1/pgm' , num )
    num += 1
