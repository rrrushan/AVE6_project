#!/usr/bin/env python3



import ros_compatibility as roscomp
from ros_compatibility.node import CompatibleNode
from ros_compatibility.qos import QoSProfile, DurabilityPolicy
import rospy


class GenerateTraffic(CompatibleNode):
    def __init__(self):
        super(GenerateTraffic, self).__init__("Traffic Generator")
        self.host = self.get_param("host", "127.0.0.1")
        self.port = int(self.get_param("port", "2000"))
        self.n = int(self.get_param("n", "30"))
        self.w = int(self.get_param("w", "30"))
        self.safe = bool(self.get_param("safe", "True"))
        self.filterv = self.get_param("filterv", "vehicle.*")
        self.generationv = self.get_param("generationv", "G")
        self.tm_port = int(self.get_param("tm-port", "8000"))
        self.asynch = bool(self.get_param("asynch", "False"))
        self.hybrid = bool(self.get_param("hybrid", "False"))
        self.seed = self.get_param("seed", "None")
        self.seedw = int(self.get_param("seedw", "0"))
        self.car_lights_on = bool(self.get_param("car-lights-on", "False"))
        self.hero = bool(self.get_param("hero", "False"))
        self.respawn = bool(self.get_param("respawn", "False"))
        self.no_rendering = bool(self.get_param("no-rendering", "False"))
def main():
    args = GenerateTraffic()
    
    if not args.asynch:
        print("True")
    else:
        print("False")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        print('\ndone.')
