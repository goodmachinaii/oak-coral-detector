from __future__ import annotations
import depthai as dai
from .config import Settings


def build_pipeline(settings: Settings):
    p = dai.Pipeline()
    cam_rgb = p.create(dai.node.ColorCamera)
    cam_rgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
    cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam_rgb.setInterleaved(False)
    cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    cam_rgb.setPreviewSize(*settings.rgb_preview_size)
    cam_rgb.setFps(settings.rgb_fps)

    xout_rgb = p.create(dai.node.XLinkOut)
    xout_rgb.setStreamName('rgb')
    xout_rgb.input.setBlocking(False)
    xout_rgb.input.setQueueSize(1)
    cam_rgb.preview.link(xout_rgb.input)

    mono_l = p.create(dai.node.MonoCamera)
    mono_r = p.create(dai.node.MonoCamera)
    mono_l.setBoardSocket(dai.CameraBoardSocket.CAM_B)
    mono_r.setBoardSocket(dai.CameraBoardSocket.CAM_C)
    mono_l.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    mono_r.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    mono_l.setFps(settings.rgb_fps)
    mono_r.setFps(settings.rgb_fps)

    stereo = p.create(dai.node.StereoDepth)
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
    stereo.setLeftRightCheck(True)
    mono_l.out.link(stereo.left)
    mono_r.out.link(stereo.right)

    xout_depth = p.create(dai.node.XLinkOut)
    xout_depth.setStreamName('depth')
    xout_depth.input.setBlocking(False)
    xout_depth.input.setQueueSize(1)
    stereo.depth.link(xout_depth.input)
    return p
