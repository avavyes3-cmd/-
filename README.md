# 森林火灾场景下的半自动跨模态配准与边缘侧实时分割系统
**Forest Fire Cross-Modal Registration & Edge-AI Real-time Segmentation System**

## 📌 项目概述
本项目针对无人机森林火灾监测场景，构建了一套从**数据生产**到**边缘部署**的全栈闭环系统。核心解决了跨模态图像（红外-可见光）标注成本高、配准鲁棒性差以及 Transformer 架构模型在边缘硬件上推理缓慢的工程痛点。



## 🚀 核心技术架构

### 1. 算法层：质量感知回退流水线 (Gated Fallback Pipeline)
[span_0](start_span)提出了一种面向离线数据生产的半自动配准与标签迁移方案[span_0](end_span)：
* **[span_1](start_span)[span_2](start_span)ECC 精配准**：作为主优化器提供 6-DOF 仿射变换校正[span_1](end_span)[span_2](end_span)。
* **[span_3](start_span)[span_4](start_span)MI 门控回退**：当 ECC 发散或检测到异常时自动触发互信息优化（2-DOF），有效配准率从 **78% 提升至 92%**[span_3](end_span)[span_4](end_span)。
* **[span_5](start_span)标签闭环迁移**：利用配准矩阵将可见光 Polygon 标注自动回投至红外坐标系，标注成本显著降低约 **36%–45%**[span_5](end_span)。
* **[span_6](start_span)单次插值优化**：合并 Warp 与 Resize 矩阵，仅执行一次 LANCZOS4 插值，完整保留可见光图像细节[span_6](end_span)。

### 2. 部署层：边缘算力板 (BPU) 深度调优
[span_7](start_span)[span_8](start_span)针对 **RDK X5 (Bayes-e BPU)** 平台，对自有轻量级 Transformer 架构模型进行极致性能压榨[span_7](end_span)[span_8](end_span)：
* **[span_9](start_span)[span_10](start_span)Softmax INT16 优化**：针对 YOLO11 中的 C2PSA 注意力模块，将 Softmax 算子配置为 **INT16 精度**，避免计算图碎片化引发的 CPU 回退[span_9](end_span)[span_10](end_span)。
* **[span_11](start_span)单子图编译**：成功实现全 BPU 运行，推理耗时从 36.13 ms 锐减至 **9.69 ms**，端到端帧率提升约 2.4 倍达到 **49 FPS**[span_11](end_span)。
* **[span_12](start_span)[span_13](start_span)NV12 直通输入**：利用板端 NV12 格式直接消费编解码器输出，单帧节省约 **1.6ms** 的色彩空间转换耗时[span_12](end_span)[span_13](end_span)。

### 3. 硬件控制层：大脑与小脑的物理闭环
* **感知大脑 (RDK X5)**：负责高层逻辑与实时感知，通过摄像头捕捉火情并解析目标像素坐标。
* **执行小脑 (STM32)**：负责底层控制，通过 **UART 串口**接收来自大脑的指令包，并转化为物理执行动作。
* **物理集成 (JLCEDA)**：基于**嘉立创 EDA** 设计专用母板，实现 12V 转 5V/3.3V 稳压及全链路硬件通讯闭环。



## 📊 实验结果与 KPI

| 指标 | 性能表现 | 数据口径 |
| :--- | :--- | :--- |
| **有效配准率 (EdgeF1@3 > 0.3)** | **92%** | [span_14](start_span)[span_15](start_span)基于 3039 对生产数据验证[span_14](end_span)[span_15](end_span) |
| **分割精度 (mAP@0.5)** | **0.848** | [span_16](start_span)针对火、烟、消防员三类目标[span_16](end_span) |
| **BPU 推理 FPS** | **49.3 FPS** | [span_17](start_span)640×640 分辨率，单 BPU 子图[span_17](end_span) |
| **端到端延迟** | **20.3 ms** | [span_18](start_span)涵盖预处理、推理与后处理全链路[span_18](end_span) |

## 📂 仓库结构规范 (项目驱动)
* [span_19](start_span)`Algorithms/` : 存放 ECC/MI 配准核心算子及标签迁移自动化脚本[span_19](end_span)。
* [span_20](start_span)`Edge-AI/` : 存放 RDK X5 量化配置文件 (.yaml)、Softmax 精度策略及推理 Demo[span_20](end_span)。
* `Embedded-Control/` : 存放 STM32 底层控制逻辑及 UART 通讯协议解析代码。
* `Hardware-PCB/` : 存放嘉立创 EDA 原理图文件及 PCB 布局参考。
* `Research-Assets/` : 存放论文各版本草稿、实验数据统计表及核心参考文献。

## 🛠️ 下一步路线图
1.  **量化实测**：完成 Transformer 架构在 BPU 上的 INT16 精度提档验证。
2.  **物理对接**：打通 RDK X5 串口发送与 STM32 接收解析的数据链路。
3.  **电路打样**：在嘉立创完成扩展底板的电路设计与物理打样。

---

