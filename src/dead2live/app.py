"""Gradio UI: image/text -> controllable 2D digital human.

Type an instruction (Chinese or English) and the character performs it:
expressions (happy / sad / angry / surprised ...), actions (blink / nod /
shake / look ...) and lip-synced talking from quoted text.
"""
from __future__ import annotations

import os
import tempfile

import cv2
import numpy as np

from .rig import CartoonRigDetector, Rig, draw_rig_overlay
from .animator import PuppetAnimator, AnimationState
from .pipeline import build_animator
from . import brain, brain_gemini, render

DEFAULT_IMAGE = "test_image.png"
FPS = 25
GEMINI_ON = brain_gemini.enable()    # True if GEMINI_API_KEY is configured

EXAMPLES = [
    "微笑",
    "非常生气并摇头",
    "先开心地笑一下，然后突然很惊讶地张大嘴巴",
    '跟我打个招呼，说"大家好，我是莎士比亚"，然后点点头',
    "你看起来有点累，打个哈欠然后摇摇头",
    "我讲了个冷笑话，你做出无语又有点生气的表情",
    "向左看看，再向右看看，最后开心地眨眨眼",
    "thinking hard, then nod slowly as if you understood",
]


class Engine:
    """Caches the routed animator + pipeline info for the current portrait."""
    def __init__(self):
        self.animator = None
        self.info = None
        self.key = None

    def set_image(self, image_rgb: np.ndarray):
        bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        key = (bgr.shape, int(bgr[::37, ::37].sum()))
        if key != self.key:
            self.animator, self.info = build_animator(bgr)
            self.key = key
        return self.animator


ENGINE = Engine()


def _load_default_rgb() -> np.ndarray:
    bgr = cv2.imread(DEFAULT_IMAGE)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def generate(image_rgb, instruction, use_llm=True):
    if image_rgb is None:
        image_rgb = _load_default_rgb()
    anim = ENGINE.set_image(image_rgb)
    # toggle the Gemini hook for this request
    brain.llm_parse_script = brain_gemini.parse_script if (use_llm and GEMINI_ON) else None
    states, plans = brain.animate(instruction or "neutral", FPS)
    frames = render.render_frames(anim, states)
    out = os.path.join(tempfile.gettempdir(), "dead2live_out.mp4")
    render.save_mp4(frames, out, FPS)
    src = "Gemini" if (use_llm and GEMINI_ON and brain.llm_parse_script) else "规则解析"
    beats = " → ".join(
        f"`{p.base}`{('+' + '/'.join(p.actions)) if p.actions else ''}"
        f"({p.duration:.1f}s)" + (f' 说:“{p.speak_text}”' if p.speak_text else '')
        for p in plans)
    pi = ENGINE.info
    pipe = (f"🧩 pipeline: 检测=`{pi.detector}` · 渲染=`{pi.renderer}` · "
            f"{'扁平插画' if pi.flat else '照片/复杂'} (score {pi.score:.2f})  \n") if pi else ""
    info = f"{pipe}**[指令 · {src}]** {len(plans)} 段 · {beats}"
    return out, info


def preview_rig(image_rgb):
    """Show the routed detector's rig overlay so the user can sanity-check it."""
    if image_rgb is None:
        image_rgb = _load_default_rgb()
    ENGINE.set_image(image_rgb)
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    ov = draw_rig_overlay(bgr, ENGINE.info.rig)
    return cv2.cvtColor(ov, cv2.COLOR_BGR2RGB)


def build_ui():
    import gradio as gr
    with gr.Blocks(title="dead2live · 图/文生数字人") as demo:
        gr.Markdown(
            "# 🎭 dead2live — 图/文生可操控数字人\n"
            "上传**任意 2D 肖像**（卡通 / 动漫 / 绘画 / 真人照片均可），输入指令，"
            "让角色做出表情和动作。支持中文/英文。\n\n"
            "系统会**自动判别画风并路由**：扁平插画→程序化重绘，照片/复杂→网格形变；"
            "检测按 CV→MediaPipe→Gemini Vision 逐级兜底。\n\n"
            "**示例指令**：`微笑` · `非常生气并摇头` · `惊讶地张大嘴` · "
            '`说话: "你好"` · `向左看然后点头`'
        )
        with gr.Row():
            with gr.Column(scale=1):
                img = gr.Image(label="肖像图片 (Portrait)", type="numpy",
                               value=_load_default_rgb())
                instr = gr.Textbox(label="指令 (Instruction)", value="微笑开心",
                                   placeholder='例如：先笑一下再惊讶地张大嘴 / 说"你好"')
                use_llm = gr.Checkbox(
                    value=GEMINI_ON,
                    label=("🤖 用 Gemini 理解自由指令（支持多段/口语）"
                           if GEMINI_ON else
                           "🤖 Gemini 未配置（用规则解析）"),
                    interactive=GEMINI_ON)
                with gr.Row():
                    go = gr.Button("🎬 生成动画", variant="primary")
                    rigbtn = gr.Button("🔍 检测预览")
                gr.Examples(examples=[[e] for e in EXAMPLES], inputs=[instr])
            with gr.Column(scale=1):
                vid = gr.Video(label="数字人动画 (Result)", autoplay=True, loop=True)
                info = gr.Markdown()
                rigview = gr.Image(label="特征检测 (Rig detection)", visible=True)

        go.click(generate, [img, instr, use_llm], [vid, info])
        rigbtn.click(preview_rig, [img], [rigview])
    return demo


def main():
    import gradio as gr
    build_ui().launch(server_name="127.0.0.1", inbrowser=True, theme=gr.themes.Soft())


if __name__ == "__main__":
    main()
