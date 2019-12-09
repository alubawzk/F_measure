import os
def ListFilesToTxt(dir,file,wildcard,recursion):
    file_name = []
    exts = wildcard.split(" ")
    files = os.listdir(dir)
    for name in files:
        fullname=os.path.join(dir,name)
        if(os.path.isdir(fullname) & recursion):
            ListFilesToTxt(fullname,file,wildcard,recursion)
        else:
            for ext in exts:
                if(name.endswith(ext)):
                    # file.write(name + "\n")
                    # file.write(name)
                    file_name.append(name)
                    break
    return file_name

def Test():
  dir="images/"
  outfile="binaries.txt"
  wildcard = ".txt .jpg .bgm .png"
 
  file = open(outfile,"w")
  if not file:
    print ("cannot open the file %s for writing" % outfile)

  ListFilesToTxt(dir,file,wildcard, 1)
 
  file.close()