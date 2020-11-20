import sys

keep = set()

fin = open(sys.argv[1], 'r')
status = True
while status:
    line = fin.readline()
    if line:
        keep.add(line.strip())
    else:
        status = False
fin.close()

nmatches = len(keep)

fin = open(sys.argv[2], 'r')
fout = open(sys.argv[3], 'w')
status = True
while status and nmatches > 0:
    line = fin.readline()
    if line:
        name = line.split()[0]
        if name in keep:
            line = line.replace(name, "%s\t1"%name, 1)
            keep.remove(name) # deswelling set strategy
            nmatches -= 1 # if all matches are found, you can also stop
        else:
            line = line.replace(name, "%s\t0"%name, 1)
            fout.write(line)
    else:
        status = True

fin.close()
fout.close()
