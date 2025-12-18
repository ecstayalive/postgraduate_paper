#!/usr/bin/env python3
import math

import numpy as np
import rclpy
from geometry_msgs.msg import Point
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import ColorRGBA
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from visualization_msgs.msg import Marker, MarkerArray


# ==========================================
# 1. 核心数学层：热力图着色器
# ==========================================
class ColorMapper:
    """提供类似 Matplotlib 'Jet' 的热力图配色，用于专业显示高度场"""

    @staticmethod
    def get_color(value, vmin, vmax):
        """将数值映射为 RGBA 颜色 (Blue -> Green -> Red)"""
        c = ColorRGBA(a=1.0)
        # 归一化
        value = np.clip(value, vmin, vmax)
        norm = (value - vmin) / (vmax - vmin + 1e-6)

        # 简单的 Jet Colormap 近似实现
        # 0.0(Blue) -> 0.25(Cyan) -> 0.5(Green) -> 0.75(Yellow) -> 1.0(Red)
        if norm < 0.25:
            c.r = 0.0
            c.g = 4.0 * norm
            c.b = 1.0
        elif norm < 0.5:
            c.r = 0.0
            c.g = 1.0
            c.b = 1.0 - 4.0 * (norm - 0.25)
        elif norm < 0.75:
            c.r = 4.0 * (norm - 0.5)
            c.g = 1.0
            c.b = 0.0
        else:
            c.r = 1.0
            c.g = 1.0 - 4.0 * (norm - 0.75)
            c.b = 0.0
        return c


# ==========================================
# 2. 地形数据层：支持双线性插值
# ==========================================
class TerrainLayer:
    def __init__(self, width=10.0, height=10.0, resolution=0.1):
        self.w = width
        self.h = height
        self.res = resolution
        self.cols = int(width / resolution)
        self.rows = int(height / resolution)

        # 存储实际的高度数据 (GridMap Data)
        self.data = np.zeros((self.cols, self.rows))

        # 记录地形的极值，用于可视化归一化
        self.min_z = 0.0
        self.max_z = 0.0

        self._generate_synthetic_terrain()

    def _generate_synthetic_terrain(self):
        """生成模拟的 Elevation Map 数据"""
        print(f"Generating Grid Map: {self.cols}x{self.rows} cells...")

        # 使用 numpy 网格加速计算
        x = np.linspace(-self.w / 2, self.w / 2, self.cols)
        y = np.linspace(-self.h / 2, self.h / 2, self.rows)
        X, Y = np.meshgrid(x, y, indexing="ij")

        # 模拟复杂地形 (类似于 elevation_mapping 接收到的传感器数据)
        # 包含：低频起伏 + 高频噪声
        Z = np.sin(X * 0.8 + Y * 0.4) * 0.4  # 基础山丘
        Z += np.cos(X * 1.5 - Y * 1.2) * 0.2  # 细节纹理
        Z += np.exp(-((X - 2) ** 2 + (Y - 2) ** 2)) * 0.5  # 局部凸起障碍物
        Z += np.random.normal(0, 0.01, (self.cols, self.rows))  # 传感器噪声

        self.data = Z
        self.min_z = np.min(self.data)
        self.max_z = np.max(self.data)
        print("Grid Map Ready.")

    def get_height_bilinear(self, x, y):
        """
        【专业采样】双线性插值 (Bilinear Interpolation)
        这是 Elevation Map 标准采样方式，比最近邻更平滑。
        """
        # 1. 转换到 Grid 坐标系 (浮点数索引)
        grid_x = (x + self.w / 2) / self.res
        grid_y = (y + self.h / 2) / self.res

        # 2. 找到四周的四个整数格点
        x0 = int(np.floor(grid_x))
        x1 = x0 + 1
        y0 = int(np.floor(grid_y))
        y1 = y0 + 1

        # 3. 边界检查
        if x0 < 0 or x1 >= self.cols or y0 < 0 or y1 >= self.rows:
            return 0.0  # Out of map

        # 4. 获取四个角的高度
        h00 = self.data[x0, y0]
        h10 = self.data[x1, y0]
        h01 = self.data[x0, y1]
        h11 = self.data[x1, y1]

        # 5. 计算权重 (在格子内部的归一化坐标)
        u = grid_x - x0
        v = grid_y - y0

        # 6. 双线性插值公式
        # f(x,y) = (1-u)(1-v)h00 + u(1-v)h10 + (1-u)vh01 + uvh11
        height = (1 - u) * (1 - v) * h00 + u * (1 - v) * h10 + (1 - u) * v * h01 + u * v * h11

        return height


# ==========================================
# 3. 可视化工具层
# ==========================================
class MarkerVisualizer:
    def __init__(self, node_clock):
        self.clock = node_clock
        self._markers = MarkerArray()
        self._id_counter = 0

    def reset(self):
        self._markers = MarkerArray()
        self._id_counter = 0

    def get_markers(self):
        return self._markers

    def _get_next_id(self):
        self._id_counter += 1
        return self._id_counter - 1

    def add_elevation_mesh(self, terrain: TerrainLayer, frame_id, z_offset=0.0):
        """绘制专业的高程网格 Mesh"""
        m = Marker()
        m.header.frame_id = frame_id
        m.header.stamp = self.clock.now().to_msg()
        m.ns = "elevation_map"
        m.id = self._get_next_id()
        m.type = Marker.TRIANGLE_LIST
        m.action = Marker.ADD
        m.pose.orientation.w = 1.0
        m.pose.position.z = z_offset
        m.scale.x = 1.0
        m.scale.y = 1.0
        m.scale.z = 1.0
        m.color.a = 1.0  # 如果不设置顶点颜色，这将是基础颜色

        # 缓存局部变量加速
        res = terrain.res
        width_offset = terrain.w / 2
        height_offset = terrain.h / 2
        data = terrain.data

        # 预计算所有顶点的物理坐标和颜色，可以显著提高大型网格的生成速度
        # 但为了代码清晰，这里仍使用循环，但在 C++ 中应当使用 Vertex Buffer
        for i in range(terrain.cols - 1):
            for j in range(terrain.rows - 1):
                # 计算四个顶点的物理坐标 (Grid Center)
                x0 = i * res - width_offset
                y0 = j * res - height_offset
                x1 = (i + 1) * res - width_offset
                y1 = (j + 1) * res - height_offset

                z00 = data[i, j]
                z10 = data[i + 1, j]
                z01 = data[i, j + 1]
                z11 = data[i + 1, j + 1]

                # 创建点
                p00 = Point(x=x0, y=y0, z=z00)
                p10 = Point(x=x1, y=y0, z=z10)
                p01 = Point(x=x0, y=y1, z=z01)
                p11 = Point(x=x1, y=y1, z=z11)

                # 获取热力图颜色
                c00 = ColorMapper.get_color(z00, terrain.min_z, terrain.max_z)
                c10 = ColorMapper.get_color(z10, terrain.min_z, terrain.max_z)
                c01 = ColorMapper.get_color(z01, terrain.min_z, terrain.max_z)
                c11 = ColorMapper.get_color(z11, terrain.min_z, terrain.max_z)

                # 添加两个三角形 (带颜色梯度的 Mesh)
                m.points.extend([p00, p10, p01])
                m.colors.extend([c00, c10, c01])

                m.points.extend([p10, p11, p01])
                m.colors.extend([c10, c11, c01])

        self._markers.markers.append(m)

    # --- 其他基础辅助函数保持不变 ---
    def add_sphere(self, frame_id, pos, color=(1.0, 0.0, 0.0, 1.0), scale=0.1):
        m = self._create_base_marker(frame_id, Marker.SPHERE)
        m.pose.position.x, m.pose.position.y, m.pose.position.z = (
            float(pos[0]),
            float(pos[1]),
            float(pos[2]),
        )
        m.scale.x = m.scale.y = m.scale.z = scale
        m.color.r, m.color.g, m.color.b, m.color.a = color
        self._markers.markers.append(m)

    def add_text(self, frame_id, pos, text, color=(0.0, 0.0, 0.0, 1.0), scale=0.1):
        m = self._create_base_marker(frame_id, Marker.TEXT_VIEW_FACING)
        m.pose.position.x, m.pose.position.y, m.pose.position.z = (
            float(pos[0]),
            float(pos[1]),
            float(pos[2]),
        )
        m.text = text
        m.scale.z = scale
        m.color.r, m.color.g, m.color.b, m.color.a = color
        self._markers.markers.append(m)

    def add_axes(self, frame_id, origin, scale=0.25, axis_width=0.02):
        x, y, z = origin
        self._add_arrow(frame_id, [x, y, z], [x + scale, y, z], (1.0, 0.0, 0.0), axis_width)
        self._add_arrow(frame_id, [x, y, z], [x, y + scale, z], (0.0, 1.0, 0.0), axis_width)
        self._add_arrow(frame_id, [x, y, z], [x, y, z + scale], (0.0, 0.0, 1.0), axis_width)

    def _add_arrow(self, frame_id, start, end, rgb, width):
        m = self._create_base_marker(frame_id, Marker.ARROW)
        m.points = [
            Point(x=float(start[0]), y=float(start[1]), z=float(start[2])),
            Point(x=float(end[0]), y=float(end[1]), z=float(end[2])),
        ]
        m.scale.x, m.scale.y, m.scale.z = width, width * 2.0, 0.0
        m.color.r, m.color.g, m.color.b, m.color.a = rgb[0], rgb[1], rgb[2], 1.0
        self._markers.markers.append(m)

    def _create_base_marker(self, frame_id, m_type):
        m = Marker()
        m.header.frame_id = frame_id
        m.header.stamp = self.clock.now().to_msg()
        m.ns = "custom_shapes"
        m.id = self._get_next_id()
        m.type = m_type
        m.action = Marker.ADD
        m.pose.orientation.w = 1.0
        return m


# ==========================================
# 4. 主节点
# ==========================================
class RobotControllerAndVisualizer(Node):
    def __init__(self):
        super().__init__("robot_controller_visualizer")

        self.joint_pub = self.create_publisher(JointState, "joint_states", 10)
        self.marker_pub = self.create_publisher(MarkerArray, "visualization_marker_array", 10)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # 使用专业版的地形层
        self.terrain = TerrainLayer(width=10.0, height=10.0, resolution=0.2)
        self.viz = MarkerVisualizer(self.get_clock())

        self.init_joint_data()
        self.timer = self.create_timer(0.1, self.timer_callback)
        self.time_counter = 0.0

        self.get_logger().info("节点已启动：专业高程图模式")

    def init_joint_data(self):
        """初始化关节名称和默认位置 (保持不变)"""
        self.joint_names = [
            "FL_hip_joint",
            "FL_thigh_joint",
            "FL_calf_joint",
            "FR_hip_joint",
            "FR_thigh_joint",
            "FR_calf_joint",
            "RL_hip_joint",
            "RL_thigh_joint",
            "RL_calf_joint",
            "RR_hip_joint",
            "RR_thigh_joint",
            "RR_calf_joint",
            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
            "joint6",
            "jointGripper",
        ]
        self.joint_positions = [
            0.1,
            -0.773,
            1.501,
            -0.1,
            -0.773,
            1.501,
            0.1,
            -0.773,
            1.501,
            -0.1,
            -0.773,
            1.501,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]

    def timer_callback(self):
        current_time = self.get_clock().now().to_msg()
        self.time_counter += 0.1

        self.viz.reset()

        # 1. 绘制专业高程图 Mesh (带热力图颜色)
        terrain_frame = "trunk"
        terrain_z_offset = -0.5
        self.viz.add_elevation_mesh(self.terrain, terrain_frame, z_offset=terrain_z_offset)

        # 2. 采样计算 (升级为双线性插值)
        sample_x = 3.0 * math.sin(self.time_counter * 0.5)
        sample_y = 3.0 * math.cos(self.time_counter * 0.5)

        # 【关键】使用 get_height_bilinear 获取精确高度
        # 地形显示的 z_offset 必须在采样后加上，或者采样前处理，这里我们在显示时体现差异
        raw_height = self.terrain.get_height_bilinear(sample_x, sample_y)
        display_z = raw_height + terrain_z_offset

        # 3. 绘制采样标记
        self.viz.add_sphere(
            terrain_frame, [sample_x, sample_y, display_z], color=(1.0, 0.0, 1.0, 1.0), scale=0.2
        )  # 紫色球
        self.viz.add_text(
            terrain_frame, [sample_x, sample_y, display_z + 0.3], f"Val: {raw_height:.3f}"
        )

        # 4. TF 和 坐标系绘制 (保持不变)
        transform = self.get_tf_transform(target_frame="trunk", source_frame="link00")
        if transform:
            tx = transform.transform.translation.x
            ty = transform.transform.translation.y
            tz = transform.transform.translation.z

            self.viz.add_axes("trunk", [tx, ty, tz])
            self.viz.add_axes("trunk", [tx, ty, tz - 0.57])
            self.viz.add_text("trunk", [tx, ty, tz + 0.29], "Mounting_Frame")
            self.viz.add_text("trunk", [tx, ty + 0.45, tz - 0.57], "Horizon_Frame")

        self.marker_pub.publish(self.viz.get_markers())

        # 5. 发布关节
        joint_msg = JointState()
        joint_msg.header.stamp = current_time
        joint_msg.name = self.joint_names
        joint_msg.position = self.joint_positions
        self.joint_pub.publish(joint_msg)

    def get_tf_transform(self, target_frame, source_frame):
        try:
            t = self.tf_buffer.lookup_transform(target_frame, source_frame, rclpy.time.Time())
            return t
        except TransformException as ex:
            self.get_logger().debug(f"TF Error: {ex}")


def main(args=None):
    rclpy.init(args=args)
    node = RobotControllerAndVisualizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

