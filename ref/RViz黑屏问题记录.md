# RViz 黑屏问题记录

本文记录 RViz 在 Docker、远程桌面、X11 转发或虚拟化图形环境中出现 3D 渲染区域纯黑时的通用排查和解决方法。

## 1. 典型现象

RViz 能正常打开，左侧 Displays 面板、菜单栏、工具栏都能显示，但中间 3D RenderPanel 是纯黑的。

常见表现：

- Grid 不显示。
- 背景颜色设置不生效。
- PointCloud2、Marker、RobotModel 等显示项可能没有明显报错，但主窗口依然黑。
- Image 面板有时也可能是黑的。
- 降低 RViz OpenGL 版本后仍然黑。

如果只加载一个只包含 Grid 的极简 RViz 配置也仍然黑，基本可以判断问题在 RViz/Qt/OGRE/OpenGL 渲染层，而不是 ROS topic、TF 或业务代码。

## 2. 先做最小化判断

新建或使用一个只显示 Grid 的 RViz 配置进行测试。这个测试不依赖任何 ROS topic。

启动 RViz：

```bash
source /opt/ros/noetic/setup.bash
rviz
```

在 RViz 中只保留：

```text
Global Options:
  Fixed Frame: world

Displays:
  Grid
```

如果此时 3D 区域仍然纯黑，说明问题不在具体项目或仿真程序。

## 3. 常规 OpenGL 兼容参数

先尝试 RViz 的 OpenGL 2.1 兼容模式：

```bash
QT_X11_NO_MITSHM=1 \
rviz --opengl 210 --disable-anti-aliasing
```

如果仍然黑，可以尝试 OpenGL 1.2：

```bash
QT_X11_NO_MITSHM=1 \
rviz --opengl 120 --disable-anti-aliasing
```

其中：

- `QT_X11_NO_MITSHM=1`：避免 Docker/X11 共享内存相关问题。
- `--opengl 210`：强制 RViz 使用 OpenGL 2.1 兼容模式。
- `--disable-anti-aliasing`：关闭抗锯齿，减少 OGRE 初始化问题。

## 4. 强制 Mesa llvmpipe 软件渲染

有些环境中，仅设置：

```bash
LIBGL_ALWAYS_SOFTWARE=1
```

并不一定真正生效。可以用 `glxinfo -B` 检查：

```bash
LIBGL_ALWAYS_SOFTWARE=1 glxinfo -B
```

如果输出仍然是 NVIDIA，例如：

```text
OpenGL vendor string: NVIDIA Corporation
OpenGL renderer string: NVIDIA ...
```

说明 GLVND 仍然选择了 NVIDIA OpenGL。此时需要强制使用 Mesa：

```bash
QT_X11_NO_MITSHM=1 \
__GLX_VENDOR_LIBRARY_NAME=mesa \
MESA_LOADER_DRIVER_OVERRIDE=llvmpipe \
GALLIUM_DRIVER=llvmpipe \
LIBGL_ALWAYS_SOFTWARE=1 \
rviz --opengl 210 --disable-anti-aliasing
```

验证是否真正切到软件渲染：

```bash
__GLX_VENDOR_LIBRARY_NAME=mesa \
MESA_LOADER_DRIVER_OVERRIDE=llvmpipe \
GALLIUM_DRIVER=llvmpipe \
LIBGL_ALWAYS_SOFTWARE=1 \
glxinfo -B
```

正常应看到类似：

```text
OpenGL vendor string: Mesa/X.org
OpenGL renderer string: llvmpipe
```

如果看到 `llvmpipe`，说明 RViz 正在使用 CPU 软件渲染。

## 5. 可能需要的 Mesa 工具

如果缺少 `glxinfo` 或 Mesa 软件渲染库，可以安装：

```bash
sudo apt-get update
sudo apt-get install -y mesa-utils libgl1-mesa-dri libglx-mesa0
```

在 Docker 镜像中也应安装这些包。

## 6. Docker/X11 相关建议

如果 RViz 在容器中运行，建议启动容器时至少保证：

```bash
-e DISPLAY=$DISPLAY
-v /tmp/.X11-unix:/tmp/.X11-unix
```

如果使用宿主机 X server，还可能需要在宿主机执行：

```bash
xhost +local:docker
```

如果硬件 OpenGL 在容器里不稳定，优先使用 Mesa llvmpipe 启动 RViz。虽然性能较低，但用于调试通常足够。

## 7. 如果仍然黑屏

如果以下条件都满足：

- 只显示 Grid 的 RViz 仍然黑。
- `--opengl 210` 和 `--opengl 120` 都无效。
- Mesa llvmpipe 已确认生效，但 RViz 仍然黑。

则通常说明当前图形转发链路与 RViz/OGRE 不兼容。可选方案：

- 在宿主机直接运行 RViz，容器内只运行 ROS 节点。
- 使用 VNC/noVNC 桌面环境运行 RViz。
- 使用 VirtualGL 或 x11docker 管理 GUI 容器。
- 换成原生 Ubuntu X11 会话，避免 Wayland/远程桌面/虚拟化叠加。

## 8. 常用清理命令

结束 RViz：

```bash
pkill -f rviz
```

如果只想检查是否还有 RViz 进程：

```bash
pgrep -af rviz
```
