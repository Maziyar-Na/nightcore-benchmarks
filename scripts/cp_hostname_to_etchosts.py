import os

with open("/etc/hosts", 'r') as inf:
    with open("temp_etchosts", 'w') as outf:
        for line in inf:
            if c == 0:
                outf.write(line[:len(line) - 1] + " " + os.system('hostname'))
            else:
                outf.write(line)

