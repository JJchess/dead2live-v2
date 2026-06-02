# dead2live · 图/文生可操控数字人

把一张**肖像图片**（或一段**文字描述**）变成一个可以用**文字指令操控**的 2D 数字人。
输入「微笑」「非常生气并摇头」「说话: "你好"」，角色就会做出对应的表情和动作。

> 当前为 **Phase 1（2D）**：在普通 CPU / 8GB 显卡上即可实时运行，无需大模型。
> Phase 2/3 会加入神经渲染（真人照片）、文生图、语音口型同步、以及 3D。

![pipeline](outputs/expressions_grid.png)

---

## 架构 (Architecture)

```
文字指令 ──►【Brain】意图解析 ──► Plan(表情 + 动作 + 时长)
                                      │
                              build_timeline ──► [AnimationState × N 帧]
图片 ──►【Rig】特征检测 ──► 特征装配(眼/瞳/眉/嘴/头) + 调色板
                                      │
                            【PuppetAnimator】程序化重绘 + 头部仿射 ──► 帧
                                      │
                              【Render】 ──► MP4 / GIF / 实时预览
```

**为什么用「程序化重绘」而不是神经网络？**
测试用的莎士比亚头像是扁平插画风，MediaPipe / LivePortrait 等基于真人脸的模型**完全检测不到**它的脸。
对插画 / 表情包 / 卡通这一类风格，最稳健、最可控、且能实时的方法是：
用经典 CV 把五官「装配」出来 → 擦除生成干净底图 → 每帧按调色板**重新绘制**五官 + 头部仿射变换。
真人照片走神经渲染路线（见 Roadmap），二者共用同一个 `Rig` 接口。

### 模块
| 文件 | 职责 |
|------|------|
| `src/dead2live/rig.py` | 特征检测：`CartoonRigDetector`（扁平插画）+ `Rig` 数据模型，可存/读 JSON |
| `src/dead2live/animator.py` | `PuppetAnimator`：干净底图 + 逐帧程序化重绘五官 + 头部仿射；`AnimationState` |
| `src/dead2live/brain.py` | 指令解析（中英双语）→ 多段 `Plan` 序列 → 缓动关键帧时间线（段间承接）|
| `src/dead2live/brain_gemini.py` | **Gemini LLM** 理解自由口语指令 → 多段动画脚本（JSON schema 约束）|
| `src/dead2live/config.py` | 读取 `.env`（`GEMINI_API_KEY` 等），不硬编码密钥 |
| `src/dead2live/render.py` | 帧序列 → MP4 / GIF |
| `src/dead2live/app.py` | Gradio Web UI（含 Gemini 开关）|

---

## 安装 & 运行

```powershell
conda activate dead2live          # 已创建好的环境 (python 3.11)
# 如需重建： conda create -n dead2live python=3.11 -y && pip install -r requirements.txt

python run.py                     # 打开 http://127.0.0.1:7860
```

命令行单测：
```powershell
python -m src.dead2live.rig        test_image.png   # 检测预览 -> outputs/rig_overlay.png
python -m src.dead2live.animator                    # 表情网格 -> outputs/expressions_grid.png
python -m src.dead2live.brain                       # 指令解析自测
```

---

## 支持的指令

**表情**：微笑/开心 · 难过/伤心 · 生气/愤怒 · 惊讶/吃惊 · 害怕 · 思考 · 平静 · 眨眼(wink)
**动作**：眨眼 · 点头 · 摇头 · 向左/右/上/下看 · 说话 · 打哈欠 · 大笑
**说话口型**：`说话: "要说的内容"` —— 用引号里的文字长度驱动嘴型时长（伪音素）
**强度**：`非常/很` 加强、`稍微/有点` 减弱
英文同样可用：`very angry and shake head`, `slightly sad`, `say "hello"` …

角色始终带有「呼吸式微动 + 自动眨眼」的待机生命感；头部点头/摇头/侧倾只动头、身体与背景保持稳定。

### 🤖 Gemini 自由指令（已接入）
配置 `.env` 里的 `GEMINI_API_KEY` 后，UI 勾选「用 Gemini 理解指令」即可输入**口语化、复合**指令，
自动拆成**多段动画**。例如：
- `先开心地笑一下，然后突然很惊讶地张大嘴巴，最后摇摇头` → happy → surprised+talk → neutral+shake
- `你看起来有点累，打个哈欠然后摇摇头`
- `跟我打个招呼，说"大家好，我是莎士比亚"，然后点点头`

未配置 key 时自动回退到规则解析，离线照常工作。

---

## Roadmap

- **Phase 2 — 真实感 & 输入扩展**
  - ✅ LLM 指令理解（Gemini）+ 多段动画序列
  - ✅ 头部分层：只变换头部而非整图（更自然的点头/摇头/侧倾）
  - ⬜ 神经渲染器（LivePortrait）用于真人照片，复用 `Rig` 接口（`Animator` 已抽象）
  - ⬜ 文生图输入：SD / SDXL-Turbo（适配 8GB），实现「一段描述 → 数字人」
  - ⬜ 语音口型：edge-tts 合成 → 音素/能量 → 真·口型同步
- **Phase 3 — 3D**
  - 单图 → 3D（如 3DGS / FLAME / 可驱动头像），文字指令驱动 blendshape + 骨骼

---

## 已知限制
- 头部位移目前作用于整图（小幅度下可接受；分层在 Phase 2）。
- `CartoonRigDetector` 针对高对比扁平插画；真人/复杂背景请走 Phase 2 神经路线或手工编辑 `Rig` JSON。
