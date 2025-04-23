"""
Object Permanence Module
==================

This module adds perceived objects to Cozmo's NavMemoryMap
"""

import math
from collections import deque
from datetime import datetime
from pathlib import Path
import numpy as np
import threading
import time
from cozmo.util import Pose, degrees
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


from retico_core import AbstractModule, UpdateType, UpdateMessage
from retico_core.dialogue import GenericDictIU
from retico_core.text import SpeechRecognitionIU
from retico_vision.vision import DetectedObjectsIU
from retico_wacnlu.common import GroundedFrameIU

class CozmoObjectPermanenceModule(AbstractModule):
    @staticmethod
    def name():
        return "Cozmo Object Permanence"

    @staticmethod
    def description():
        return "A module that tracks the pose information of objects viewed by Cozmo and manages them in Cozmo's Memory Map"

    @staticmethod
    def input_ius():
        return [DetectedObjectsIU, SpeechRecognitionIU]

    @staticmethod
    def output_iu():
        return CozmoObjPermanenceIU

    def __init__(self, robot, **kwargs):
        super().__init__(**kwargs)
        self.robot = robot
        self.objects = []
        self.tracked_objects = {}
        self.top_object = None
        self.queue = deque(maxlen=1)
        self.current_behavior = None
        self.robot_start_position = robot.pose

    def process_update(self, update_message):
        for iu, ut in update_message:
            if ut != UpdateType.ADD:
                continue
            else:
                self.queue.append(iu)

    def go_to_object(self, object_name):
        object_pose = None
        if object_name in self.tracked_objects:
            seen = self.tracked_objects[object_name]
            if len(seen) > 1:
                # TODO: do more with this. Maybe list unique id and let them pick? maybe store other attributes, like color?
                logger.info(f"seen {len(seen)} objects with that name. Selecting the first (index 0)")
            object_pose = seen[0]['object_pose']
        if object_pose:
            self.stop_execution()
            # TODO: do this in a nicer way. p.s. divide by 3 is arbitrary/experimental
            near_object_pose = Pose(x=object_pose.position.x - (object_pose.position.x/3),
                                    y=object_pose.position.y,
                                    z=object_pose.position.z,
                                    q0=object_pose.rotation.q0,
                                    q1=object_pose.rotation.q1,
                                    q2=object_pose.rotation.q2,
                                    q3=object_pose.rotation.q3)

            self.robot.go_to_pose(near_object_pose).wait_for_completed()

    def calc_distance_from_cozmo(self, object_perceived_width):
        # TODO: Do more reading on this calculation -- can I do this in a meaningful way?
        cozmo_camera_focal_length = self.robot.camera.config.focal_length
        # This is fairly arbitrary. Most of the objects I was working with were ~3 inches wide
        known_width_mm = 76
        return (known_width_mm * cozmo_camera_focal_length.x) / object_perceived_width

    def begin_explore(self):
        # TODO: this would ideally be done w/ a DM but this is a quicker implementation
        if self.current_behavior is None:
            # TODO: this behaviour moves too fast so sometimes Cozmo has moved on by the time an object is detected.
            #  Can I slow down?
            self.current_behavior = self.robot.start_behavior(cozmo.behavior.BehaviorTypes.LookAroundInPlace)

    def stop_execution(self):
        # TODO: this would ideally be done w/ a DM but this is a quicker implementation
        if self.current_behavior is not None:
            self.current_behavior.stop()
            self.current_behavior = None

    def _extractor_thread(self):
        while True:
            if len(self.queue) == 0:
                time.sleep(0.5)
                continue

            input_iu = self.queue.popleft()

            # TODO: Move this logic out of the module -- use DM instead
            if isinstance(input_iu, SpeechRecognitionIU):
                logger.info(f"Input IU text: {input_iu.text}")
                input_text = input_iu.text.strip().lower()
                if "pause" in input_text:
                    self.stop_execution()
                elif "explore" in input_text:
                    # checking "explore" _in_ input_text because ASR kept picking up "explorer"
                    self.begin_explore()
                elif "home" in input_text:
                    self.robot.go_to_pose(self.robot_start_position).wait_for_completed()
                else:
                    self.go_to_object(input_text)
                # Continue top level while loop
                continue

            if isinstance(input_iu, DetectedObjectsIU):
                self.objects = input_iu.payload
                if len(self.objects) > 0:
                    # If multiple objects are detected, only keeping the top object for now
                    self.top_object = input_iu.payload['object0']
                    self.queue.append(input_iu)
            else:
                self.top_object = None

            # Get object label from input_iu
            object_label = input_iu.payload['object0']['label_str']
            # Check if we've already seen this object
            if object_label in self.tracked_objects:
                # TODO: instead of ignoring repeat objects, consider a way to update?
                logger.info(f"Object '{object_label}' already tracked")
                self.queue.clear()  # drop frames, we've seen this object # TODO: is this a bad idea?
                # Continue top level while loop
                continue
            self.stop_execution() # TODO: This isn't stopping the robot look around behaviour reliably. Figure out why
            # divide by two is pretty arbitrary. Objects were too big in 3d space otherwise.
            # TODO: track down dimensions of things so I can scale correctly (or better than arbitrary/2)
            object_x_dims = (input_iu.payload['object0']['xmax'] - input_iu.payload['object0']['xmin'])/2
            object_y_dims = (input_iu.payload['object0']['ymax'] - input_iu.payload['object0']['ymin'])/2
            distance = self.calc_distance_from_cozmo(object_x_dims)
            # If the object is very small, set to a 10x10 cube.
            # TODO: should I do this before or after distance calculation?
            if object_x_dims < 10 or object_y_dims < 10:
                object_x_dims = 10
                object_y_dims = 10
            output_iu = self.create_iu(grounded_in=input_iu)
            robot_pose = self.robot.pose
            fixed_object = self.robot.world.create_custom_fixed_object(Pose(distance, 0, object_x_dims/2, angle_z=degrees(0)),
                                                                       x_size_mm=object_y_dims, y_size_mm=object_y_dims, z_size_mm=object_x_dims, relative_to_robot=True)
            output_iu.set_object(object_name=object_label, object_id=fixed_object.object_id)
            if object_label in self.tracked_objects:
                self.tracked_objects[object_label].append({'robot_pose': robot_pose, 'object_pose': fixed_object.pose, 'object_name': object_label, 'object_id': fixed_object.object_id})
            else:
                self.tracked_objects[object_label] = [{'robot_pose': robot_pose, 'object_pose': fixed_object.pose, 'object_name': object_label, 'object_id': fixed_object.object_id}]
            self.robot.say_text(object_label, play_excited_animation=False, use_cozmo_voice=True, in_parallel=True, num_retries=1).wait_for_completed()

            um = UpdateMessage.from_iu(output_iu, UpdateType.ADD)
            self.append(um)

    def prepare_run(self):
        self._extractor_thread_active = True
        threading.Thread(target=self._extractor_thread).start()

    def shutdown(self):
        self._extractor_thread_active = False
