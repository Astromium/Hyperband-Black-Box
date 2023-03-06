import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-n', '--name', help='test')
parser.add_argument('-p', '--pe', help='test2')

args = parser.parse_args()
print(f'args {args.pe}')