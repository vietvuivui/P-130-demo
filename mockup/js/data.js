/**
 * AutoLabel 3D - Mock Data & State
 */

const APP_DATA = {
  currentUser: {
    name: "Nguyễn Văn A",
    email: "huy@example.com",
    role: "Annotator",
    avatar: "NH"
  },

  projects: [
    {
      id: "VF_Drive_01",
      name: "VF_Drive_01",
      framesCount: 12480,
      status: "Đang gán nhãn",
      statusCode: "in_progress",
      updatedAt: "19/09/2026",
      annotatedFrames: 4270,
      pendingFrames: 8210
    },
    {
      id: "Urban_Scene",
      name: "Urban_Scene",
      framesCount: 8320,
      status: "Chờ xử lý",
      statusCode: "pending",
      updatedAt: "18/09/2026",
      annotatedFrames: 0,
      pendingFrames: 8320
    },
    {
      id: "Highway_Test",
      name: "Highway_Test",
      framesCount: 5210,
      status: "Hoàn thành",
      statusCode: "completed",
      updatedAt: "16/09/2026",
      annotatedFrames: 5210,
      pendingFrames: 0
    },
    {
      id: "Custom_Data",
      name: "Custom_Data",
      framesCount: 3450,
      status: "Đang gán nhãn",
      statusCode: "in_progress",
      updatedAt: "15/09/2026",
      annotatedFrames: 1200,
      pendingFrames: 2250
    }
  ],

  frames: [
    {
      id: "scene001_0001",
      status: "Chưa gán nhãn",
      statusCode: "unannotated",
      time: "-",
      objectCount: 5,
      notes: ""
    },
    {
      id: "scene001_0002",
      status: "Đã gán nhãn",
      statusCode: "annotated",
      time: "5 phút trước",
      objectCount: 6,
      notes: "Đã tinh chỉnh box xe tải lane phải"
    },
    {
      id: "scene001_0003",
      status: "Chưa gán nhãn",
      statusCode: "unannotated",
      time: "-",
      objectCount: 4,
      notes: ""
    },
    {
      id: "scene001_0004",
      status: "Đã gán nhãn",
      statusCode: "annotated",
      time: "12 phút trước",
      objectCount: 5,
      notes: "Thời tiết nhiều nắng, camera chói nhẹ"
    },
    {
      id: "scene001_0005",
      status: "Chưa gán nhãn",
      statusCode: "unannotated",
      time: "-",
      objectCount: 7,
      notes: ""
    }
  ],

  nearbyFrames: [
    { id: "scene001_0000", label: "0000", isCurrent: false },
    { id: "scene001_0001", label: "0001", isCurrent: true },
    { id: "scene001_0002", label: "0002", isCurrent: false },
    { id: "scene001_0003", label: "0003", isCurrent: false },
    { id: "scene001_0004", label: "0004", isCurrent: false }
  ],

  objects: [
    {
      id: 1,
      name: "Car #1",
      className: "car",
      classLabel: "car",
      color: "#10b981", // green
      box2D: { x: 172, y: 462, width: 184, height: 246 },
      box3D: { x: -2.7, y: 11.5, z: -0.3, dx: 2.0, dy: 4.8, dz: 1.7, yaw: 0.04 },
      has2D: true,
      has3D: true,
      confidence: 0.92,
      occlusion: "Không bị che khuất",
      truncated: 0.0,
      subclass: "SUV / Crossover"
    },
    {
      id: 2,
      name: "Car #2",
      className: "car",
      classLabel: "car",
      color: "#ef4444", // red
      box2D: { x: 632, y: 498, width: 138, height: 165 },
      box3D: { x: 2.8, y: 15.2, z: -0.4, dx: 1.8, dy: 4.5, dz: 1.5, yaw: -0.02 },
      has2D: true,
      has3D: true,
      confidence: 0.87,
      occlusion: "Che khuất một phần",
      truncated: 0.0,
      subclass: "Sedan"
    },
    {
      id: 3,
      name: "Truck #3",
      className: "truck",
      classLabel: "truck",
      color: "#3b82f6", // blue
      box2D: { x: 768, y: 312, width: 192, height: 320 },
      box3D: { x: 5.6, y: 22.0, z: 0.8, dx: 2.6, dy: 8.4, dz: 3.4, yaw: -0.05 },
      has2D: true,
      has3D: true,
      confidence: 0.81,
      occlusion: "Rõ ràng",
      truncated: 0.05,
      subclass: "Box Truck"
    },
    {
      id: 4,
      name: "Pedestrian #4",
      className: "pedestrian",
      classLabel: "pedestrian",
      color: "#f59e0b", // yellow/amber
      box2D: { x: 52, y: 485, width: 22, height: 95 },
      box3D: { x: -7.5, y: 16.0, z: -0.2, dx: 0.7, dy: 0.7, dz: 1.75, yaw: 0.0 },
      has2D: true,
      has3D: true,
      confidence: 0.78,
      occlusion: "Bên lề đường",
      truncated: 0.0,
      subclass: "Người đi bộ"
    },
    {
      id: 5,
      name: "Bus #5",
      className: "bus",
      classLabel: "bus",
      color: "#8b5cf6", // purple
      box2D: { x: 468, y: 486, width: 104, height: 142 },
      box3D: { x: 0.1, y: 24.5, z: -0.2, dx: 2.1, dy: 4.6, dz: 1.6, yaw: 0.01 },
      has2D: true,
      has3D: false, // matches screenshot: 2D checked, 3D unchecked
      confidence: 0.65,
      occlusion: "Xa / mờ",
      truncated: 0.0,
      subclass: "Mid-size Car / Bus"
    }
  ],

  shortcuts: [
    { key: "Space", desc: "Chuyển sang frame tiếp theo" },
    { key: "Shift + Space", desc: "Quay về frame trước đó" },
    { key: "B", desc: "Công cụ vẽ Bounding Box 2D" },
    { key: "C", desc: "Công cụ tạo Bounding Cuboid 3D" },
    { key: "V", desc: "Công cụ Chọn / Di chuyển (Select)" },
    { key: "A", desc: "Kích hoạt Chạy Auto-label AI" },
    { key: "Del / Backspace", desc: "Xóa object đang chọn" },
    { key: "Enter", desc: "Xác nhận & Lưu hoàn thành frame" },
    { key: "Ctrl + Z", desc: "Hoàn tác (Undo)" },
    { key: "R", desc: "Reset góc nhìn LiDAR 3D" }
  ]
};

window.APP_DATA = APP_DATA;
