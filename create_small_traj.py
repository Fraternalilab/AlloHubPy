import glob

files = glob.glob("*sasta")

for in_file in files:
    name = "small_%s" % in_file
    out = open(name, "w")
    with open(in_file, "r") as inn:
        for line in inn:
            line = line.rstrip()
            if line.startswith(">"):
                out.write(line)
                out.write("\n")
            else:
                out.write(line[:20])
                out.write("\n")
    out.close()
