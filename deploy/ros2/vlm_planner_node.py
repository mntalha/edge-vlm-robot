"""ROS 2 node skeleton: the VLM "System 2" planner for the real robot.

Status: design skeleton (not yet tested on hardware). It shows how the code in
this repo plugs into a ROS 2 / Isaac ROS robot:

  /camera/color/image_raw (sensor_msgs/Image)  ─┐
  /voice_command          (std_msgs/String)    ─┼─► VLMPlannerNode ─► /robot/skill (std_msgs/String, JSON)
  /robot/state            (std_msgs/String)    ─┘        (room, holding - from localization + gripper)

A separate *skill executor* node (not included yet) turns each skill into
Nav2 goals (navigate/return), a grasp pipeline (pick) or an arm motion
(handover) and publishes the updated /robot/state when the skill finishes.
"""
from __future__ import annotations

import json

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String

from edge_vlm.actions import check_preconditions, parse_action
from edge_vlm.domain import build_prompt
from edge_vlm.train import VLMPolicy, load_checkpoint


class VLMPlannerNode(Node):
    def __init__(self) -> None:
        super().__init__("vlm_planner")
        self.declare_parameter("checkpoint", "checkpoints/tiny_robot_vlm.pt")
        self.declare_parameter("device", "cuda")
        self.declare_parameter("rate_hz", 1.0)  # System 2 runs slowly; controllers run fast
        ckpt = self.get_parameter("checkpoint").value
        model, tok = load_checkpoint(ckpt, device=self.get_parameter("device").value)
        self.policy = VLMPolicy(model, tok)
        self.image_size = model.cfg.image_size

        self.command = None
        self.image = None
        self.state = {"room": "living_room", "holding": None, "visited": ["living_room"], "busy": False}

        self.create_subscription(Image, "/camera/color/image_raw", self.on_image, 1)
        self.create_subscription(String, "/voice_command", self.on_command, 10)
        self.create_subscription(String, "/robot/state", self.on_state, 10)
        self.skill_pub = self.create_publisher(String, "/robot/skill", 10)
        self.create_timer(1.0 / self.get_parameter("rate_hz").value, self.tick)

    def on_image(self, msg: Image) -> None:
        # Assumes rgb8/bgr8 encoding. cv_bridge is the usual choice; numpy avoids the dependency.
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)[..., :3]
        if msg.encoding.lower().startswith("bgr"):
            img = img[..., ::-1]
        ys = np.linspace(0, msg.height - 1, self.image_size).astype(int)
        xs = np.linspace(0, msg.width - 1, self.image_size).astype(int)
        self.image = np.ascontiguousarray(img[ys][:, xs])  # nearest-neighbour resize

    def on_command(self, msg: String) -> None:
        self.command = msg.data
        self.state.update(visited=[self.state["room"]], busy=False)
        self.get_logger().info(f"new command: {self.command}")

    def on_state(self, msg: String) -> None:
        s = json.loads(msg.data)  # e.g. {"room": "kitchen", "holding": null, "busy": false}
        self.state.update(s)
        if s.get("room") and s["room"] not in self.state["visited"]:
            self.state["visited"].append(s["room"])

    def tick(self) -> None:
        if self.command is None or self.image is None or self.state["busy"]:
            return
        prompt = build_prompt(self.command, self.state["room"], self.state["holding"], self.state["visited"])
        raw = self.policy.predict(self.image, prompt)
        action = parse_action(raw)
        if action is None:
            self.get_logger().warn(f"unparseable VLM output '{raw}' - holding position")
            return
        ok, why = check_preconditions(action, self.state["room"], self.state["holding"])
        if not ok:
            self.get_logger().warn(f"guardrail rejected '{raw}': {why}")
            return
        self.state["busy"] = True  # executor clears it via /robot/state when done
        self.skill_pub.publish(String(data=action.to_json()))
        self.get_logger().info(f"skill -> {action.to_json()}")
        if action.skill == "handover":
            self.command = None


def main() -> None:
    rclpy.init()
    rclpy.spin(VLMPlannerNode())
    rclpy.shutdown()


if __name__ == "__main__":
    main()
