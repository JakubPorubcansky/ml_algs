import os
from faker import Faker

fake = Faker()
fake.seed_instance(12345)

script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, 'names.txt')

N = 20000

with open(file_path, 'w') as f:
    for i in range(N):
        first_name = fake.first_name()
        last_name = fake.last_name()
        line = first_name + ' ' + last_name
        if i < N - 1:
            line = line + '\n'
        f.write(line)
