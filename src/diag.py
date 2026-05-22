import accelerate.big_modeling as bm
import inspect
src = inspect.getsourcefile(bm)
lines = open(src).readlines()
print("Line 508-516:")
for i, l in enumerate(lines[507:516], 508):
    print(i, l, end="")