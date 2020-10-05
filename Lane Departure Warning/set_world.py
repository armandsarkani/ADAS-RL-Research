import glob
import os
import sys

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import carla
import argparse
argparser = argparse.ArgumentParser(description='Set Carla World')
argparser.add_argument(
    '-w', '--world',
    metavar='WORLD',
    default='Town06',
    help='name of carla world to switch to (e.g. Town01)')
args = argparser.parse_args()
client = carla.Client('localhost', 2000)
client.set_timeout(20)
world = client.load_world(args.world)
