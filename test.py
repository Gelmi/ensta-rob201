import time

for i in range(10):
    print(f'i: {i}', end='\r', flush=True)
    time.sleep(1)
print("A")
