import rosbag
from geometry_msgs.msg import Point
from geometry_msgs.msg import Vector3
import pandas as pd

# The bag file should be in the same directory as your terminal
bag = rosbag.Bag('./recorded-data.bag')
topic = '/planning/pos_cmd'
output_column_names = ['pos_x', 'pos_y', 'pos_z', 'vel_x', 'vel_y', 'vel_z', \
                       'acc_x', 'acc_y', 'acc_z', 'yaw', 'yaw_dot', \
                       'kx_x', 'kx_y', 'kx_z', 'kv_x', 'kv_y', 'kv_z']
output_df = pd.DataFrame(columns=column_names)

for topic, msg, t in bag.read_messages(topics=topic):

    position_x = msg.position.x
    position_y = msg.position.y
    position_z = msg.position.z

    velocity_x = msg.velocity.x
    velocity_y = msg.velocity.y
    velocity_z = msg.velocity.z

    acceleration_x = msg.acceleration.x
    acceleration_y = msg.acceleration.y
    acceleration_z = msg.acceleration.z

    yaw = msg.yaw
    yaw_dot = msg.yaw_dot

    kx_x = msg.kx[0]
    kx_y = msg.kx[1]
    kx_z = msg.kx[2]

    kv_x = msg.kv[0]
    kv_y = msg.kv[1]
    kv_z = msg.kv[2]

    output_df = output_df.append(
        {'pos_x': position_x,
         'pos_y': position_y,
         'pos_z': position_z,
         'vel_x': velocity_x,
         'vel_y': velocity_y,
         'vel_z': velocity_z,
         'acc_x': acceleration_x,
         'acc_y': acceleration_y,
         'acc_z': acceleration_z,
         'yaw': yaw,
         'yaw_dot': yaw_dot,
         'kx_x': kx_x,
         'kx_y': kx_y,
         'kx_z': kx_z,
         'kv_x': kv_x,
         'kv_y': kv_y,
         'kv_z': kv_z
        },
        ignore_index=True
    )

output_df.to_csv('output.csv')