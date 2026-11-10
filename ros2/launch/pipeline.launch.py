"""ros2 launch /app/launch/pipeline.launch.py  (settings come from environment variables)."""
import os

from launch import LaunchDescription
from launch.actions import EmitEvent, ExecuteProcess, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown


def generate_launch_description():
    video = os.environ.get("VIDEO", "/data/demo720.mp4")
    name = os.environ.get("RUN_NAME", "demo")
    hz = str(float(os.environ.get("PUBLISH_HZ", "4.0")))
    secs = str(float(os.environ.get("MAX_SECONDS", "20")))

    def node(script, *params):
        cmd = ["python3", "/app/nodes/" + script, "--ros-args"]
        for p in params:
            cmd += ["-p", p]
        return ExecuteProcess(cmd=cmd, output="screen")

    recorder = node("recorder_node.py", "name:=" + name, "fps:=" + hz)
    perception = node("perception_node.py", "name:=" + name)
    camera = node("camera_node.py", "video:=" + video, "publish_hz:=" + hz, "max_seconds:=" + secs)
    return LaunchDescription([
        recorder, perception, camera,
        RegisterEventHandler(OnProcessExit(target_action=recorder, on_exit=[EmitEvent(event=Shutdown())])),
    ])
