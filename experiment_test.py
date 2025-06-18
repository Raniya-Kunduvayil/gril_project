import tensorflow as tf
import numpy as np
import argparse

from airsim_utils import AirSimEnv
from agil_airsim import arilNN
from losses import action_loss

from ultralytics import YOLO  # for YOLO-based detection


customObjects = {
    'action_loss': action_loss
}

aril_model = "gril.h5"
aril = tf.keras.models.load_model(aril_model, custom_objects=customObjects)


env = AirSimEnv()
env.connectQuadrotor()
env.enableAPI(True)
env.armQuadrotor()
env.takeOff()
env.hover()

parser = argparse.ArgumentParser(
    prog='Experiment',
    description='Configurations for the experiment',
    epilog='Configuration include the number of episodes'
)

parser.add_argument('-e', '--episodes', type=int, help='Number of episodes to run', default=10)
parser.add_argument('-d', '--duration', type=int, help='Duration of control command', default=1)
parser.add_argument('-sc', '--sc', type=int, help='Constant related to ang->lin', default=10)

args = parser.parse_args()

# Use arguments like this:
EPISODES = args.episodes
DURATION = args.duration
SC = args.sc


for i in range(EPISODES):
    done = False
    while not done:
        img_rgb, img_depth = env.getRGBImage(), env.getDepthImage()
        commands, gaze = arilNN(img_rgb, img_depth, aril)

        # You can replace these with actual predictions from `commands` if needed
        pitch = np.random.uniform(-1, 1)
        roll = np.random.uniform(-1, 1)
        yaw = np.random.uniform(-1, 1)
        throttle = np.random.uniform(0, 1)

        vx, vy, vz = env.angularRatesToLinearVelocity(pitch, roll, yaw, throttle, SC)
        vb = env.inertialToBodyFrame(yaw, vx, vy)
        env.controlQuadrotor(vb, vz, DURATION)

        # Optional: stop after one loop to prevent infinite hovering
        done = True
