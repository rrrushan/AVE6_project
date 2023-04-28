#!/usr/bin/env python 

import rospy
from dataclasses import dataclass

from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from geometry_msgs.msg import Point
from typing import Tuple


@dataclass
class BoundBox():
    position: Tuple[float, float, float] # x, y, z
    orientation: Tuple[float, float, float, float] # x, y, z, w
    scale: Tuple[float, float, float] # x, y, z
    object_type: int

bounding_boxes = [BoundBox(position=(1, 1, 1), orientation=(0, 0, 0, 1), scale=(1, 1, 1), object_type=0),
                  BoundBox(position=(-1, 1, 1), orientation=(0, 0, 0, 1), scale=(1, 1, 1.5), object_type=1)]

def marker_pub():
    rospy.init_node('marker_pub', anonymous=True) 
    topic = 'rgb_front/visualization_marker_array' 
    frame_name = 'ego_vehicle/rgb_front' # TODO: To be checked
    pub = rospy.Publisher(topic, MarkerArray, queue_size=10)



    while not rospy.is_shutdown():
        marker_array = MarkerArray() # creating the message

        for idx, box in enumerate(bounding_boxes): 
            

            marker = Marker()
            marker.header.frame_id = frame_name
            marker.header.stamp = rospy.Time.now() # TODO: Change to timestamp from image
            marker.ns = 'bounding_box'

            marker.id = idx
            marker.type = Marker.CUBE
            marker.action = Marker.ADD

            marker.pose.position.x = box.position[0]
            marker.pose.position.y = box.position[1]
            marker.pose.position.z = box.position[2]
            marker.pose.orientation.x = box.orientation[0]
            marker.pose.orientation.y = box.orientation[1]
            marker.pose.orientation.z = box.orientation[2]
            marker.pose.orientation.w = box.orientation[3]

            marker.scale.x = box.scale[0]
            marker.scale.y = box.scale[1]
            marker.scale.z = box.scale[2]

            if box.object_type == 0:
                marker.color.r = 1.0
                marker.color.g = 0.0
                marker.color.b = 0.0
            elif box.object_type == 1:
                marker.color.r = 1.0
                marker.color.g = 0.0
                marker.color.b = 0.0
            marker.color.a = 0.5

            marker.lifetime = 0 # rospy.Duration(1.0)
            marker_array.markers.append(marker)
    
        rospy.loginfo(f'reached here \n {marker_array}')
        pub.publish(marker_array)
        
if __name__ == '__main__':
    rospy.loginfo("The file works")
    try:
        marker_pub()
    except rospy.ROSInterruptException:
        pass