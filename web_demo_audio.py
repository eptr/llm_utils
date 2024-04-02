from argparse import ArgumentParser
from pathlib import Path

import copy
import gradio as gr
import os
import re
import secrets
import tempfile
from modelscope import (
    AutoModelForCausalLM, AutoTokenizer, GenerationConfig
)
from pydub import AudioSegment


ENT_LOGOS = {
    'qwen': 'https://acd-assets.alicdn.com/acd_work/tongyi-portal/assets/logo.svg',
    'adept': 'https://www.adept.ai/images/adept-logo.png'
}

DEFAULT_CKPT_PATH = os.getenv('MODEL_NAME', default='Qwen/Qwen-Audio-Chat')
ENT_NAME = Path(DEFAULT_CKPT_PATH).parent
ENT_LOGO = ENT_LOGOS[ENT_NAME] if ENT_NAME in ENT_LOGOS else None
MDL_NAME = Path(DEFAULT_CKPT_PATH).name

def _get_args():
    parser = ArgumentParser()
    parser.add_argument("-c", "--checkpoint-path", type=str, default=DEFAULT_CKPT_PATH,
                        help="Checkpoint name or path, default to %(default)r")
    parser.add_argument("--cpu-only", action="store_true", help="Run demo with CPU only")

    parser.add_argument("--share", action="store_true", default=False,
                        help="Create a publicly shareable link for the interface.")
    parser.add_argument("--inbrowser", action="store_true", default=False,
                        help="Automatically launch the interface in a new tab on the default browser.")
    parser.add_argument("--server-port", type=int, default=8000,
                        help="Demo server port.")
    parser.add_argument("--server-name", type=str, default="127.0.0.1",
                        help="Demo server name.")

    args = parser.parse_args()
    return args


def _load_model_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint_path, trust_remote_code=True, resume_download=True,
    )

    if args.cpu_only:
        device_map = "cpu"
    else:
        device_map = "cuda"

    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint_path,
        device_map=device_map,
        trust_remote_code=True,
        resume_download=True,
    ).eval()
    model.generation_config = GenerationConfig.from_pretrained(
        args.checkpoint_path, trust_remote_code=True, resume_download=True,
    )

    return model, tokenizer


def _parse_text(text):
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split("`")
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = f"<br></code></pre>"
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace("`", r"\`")
                    line = line.replace("<", "&lt;")
                    line = line.replace(">", "&gt;")
                    line = line.replace(" ", "&nbsp;")
                    line = line.replace("*", "&ast;")
                    line = line.replace("_", "&lowbar;")
                    line = line.replace("-", "&#45;")
                    line = line.replace(".", "&#46;")
                    line = line.replace("!", "&#33;")
                    line = line.replace("(", "&#40;")
                    line = line.replace(")", "&#41;")
                    line = line.replace("$", "&#36;")
                lines[i] = "<br>" + line
    text = "".join(lines)
    return text


def _launch_demo(args, model, tokenizer):
    uploaded_file_dir = os.environ.get("GRADIO_TEMP_DIR") or str(
        Path(tempfile.gettempdir()) / "gradio"
    )

    def predict(_chatbot, task_history):
        query = task_history[-1][0]
        print("User: " + _parse_text(query))
        history_cp = copy.deepcopy(task_history)
        full_response = ""

        history_filter = []
        audio_idx = 1
        pre = ""
        global last_audio
        for i, (q, a) in enumerate(history_cp):
            if isinstance(q, (tuple, list)):
                last_audio = q[0]
                q = f'Audio {audio_idx}: <audio>{q[0]}</audio>'
                pre += q + '\n'
                audio_idx += 1
            else:
                pre += q
                history_filter.append((pre, a))
                pre = ""
        history, message = history_filter[:-1], history_filter[-1][0]
        response, history = model.chat(tokenizer, message, history=history)
        ts_pattern = r"<\|\d{1,2}\.\d+\|>"
        all_time_stamps = re.findall(ts_pattern, response)
        print(response)
        if (len(all_time_stamps) > 0) and (len(all_time_stamps) % 2 ==0) and last_audio:
            ts_float = [ float(t.replace("<|","").replace("|>","")) for t in all_time_stamps]
            ts_float_pair = [ts_float[i:i + 2] for i in range(0,len(all_time_stamps),2)]
            # 读取音频文件
            format = os.path.splitext(last_audio)[-1].replace(".","")
            audio_file = AudioSegment.from_file(last_audio, format=format)
            chat_response_t = response.replace("<|", "").replace("|>", "")
            chat_response = chat_response_t
            temp_dir = secrets.token_hex(20)
            temp_dir = Path(uploaded_file_dir) / temp_dir
            temp_dir.mkdir(exist_ok=True, parents=True)
            # 截取音频文件
            for pair in ts_float_pair:
                audio_clip = audio_file[pair[0] * 1000: pair[1] * 1000]
                # 保存音频文件
                name = f"tmp{secrets.token_hex(5)}.{format}"
                filename = temp_dir / name
                audio_clip.export(filename, format=format)
                _chatbot[-1] = (_parse_text(query), chat_response)
                _chatbot.append((None, (str(filename),)))
        else:
            _chatbot[-1] = (_parse_text(query), response)

        full_response = _parse_text(response)

        task_history[-1] = (query, full_response)
        print("Qwen-Audio-Chat: " + _parse_text(full_response))
        return _chatbot

    def regenerate(_chatbot, task_history):
        if not task_history:
            return _chatbot
        item = task_history[-1]
        if item[1] is None:
            return _chatbot
        task_history[-1] = (item[0], None)
        chatbot_item = _chatbot.pop(-1)
        if chatbot_item[0] is None:
            _chatbot[-1] = (_chatbot[-1][0], None)
        else:
            _chatbot.append((chatbot_item[0], None))
        return predict(_chatbot, task_history)

    def add_text(history, task_history, text):
        history = history + [(_parse_text(text), None)]
        task_history = task_history + [(text, None)]
        return history, task_history, ""

    def add_file(history, task_history, file):
        history = history + [((file.name,), None)]
        task_history = task_history + [((file.name,), None)]
        return history, task_history

    def add_mic(history, task_history, file):
        if file is None:
            return history, task_history
        os.rename(file, file + '.wav')
        print("add_mic file:", file)
        print("add_mic history:", history)
        print("add_mic task_history:", task_history)
        # history = history + [((file.name,), None)]
        # task_history = task_history + [((file.name,), None)]
        task_history = task_history + [((file + '.wav',), None)]
        history = history + [((file + '.wav',), None)]
        print("task_history", task_history)
        return history, task_history

    def reset_user_input():
        return gr.update(value="")

    def reset_state(task_history):
        task_history.clear()
        return []

    with gr.Blocks() as demo:
        if ENT_LOGO is not None:
            gr.Markdown(f'<p align="center"><img src="{ENT_LOGO}" style="height: 80px"/><p>')
        gr.Markdown(f'<center><font size=8>{MDL_NAME}</center>')

        chatbot = gr.Chatbot(label=MDL_NAME, elem_classes="control-height", height=750)
        query = gr.Textbox(lines=2, label='Input')
        task_history = gr.State([])
        mic = gr.Audio(sources=["microphone"], type="filepath")

        with gr.Row():
            addfile_btn = gr.UploadButton("🎤 Upload (上传)", file_types=["audio"])
            submit_btn = gr.Button("🚀 Submit (发送)")
            regen_btn = gr.Button("🔄 Retry (重试)")
            empty_bin = gr.Button("🧹 Clear (清除)")

        mic.change(add_mic, [chatbot, task_history, mic], [chatbot, task_history])
        addfile_btn.upload(add_file, [chatbot, task_history, addfile_btn], [chatbot, task_history], show_progress=True)
        submit_btn.click(add_text, [chatbot, task_history, query], [chatbot, task_history]).then(
            predict, [chatbot, task_history], [chatbot], show_progress=True
        )
        submit_btn.click(reset_user_input, [], [query])
        regen_btn.click(regenerate, [chatbot, task_history], [chatbot], show_progress=True)
        empty_bin.click(reset_state, [task_history], [chatbot], show_progress=True)

    demo.queue().launch(
        share=args.share,
        inbrowser=args.inbrowser,
        server_port=args.server_port,
        server_name=args.server_name
    )


def main():
    args = _get_args()
    model, tokenizer = _load_model_tokenizer(args)
    _launch_demo(args, model, tokenizer)


if __name__ == '__main__':
    main()
