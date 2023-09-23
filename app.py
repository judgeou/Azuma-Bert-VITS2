import sys, os

if sys.platform == "darwin":
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import logging

logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("markdown_it").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

logging.basicConfig(level=logging.INFO, format="| %(name)s | %(levelname)s | %(message)s")

logger = logging.getLogger(__name__)

import torch
import argparse
import commons
import utils
from models import SynthesizerTrn
from text.symbols import symbols
from text import cleaned_text_to_sequence, get_bert
from text.cleaner import clean_text
import gradio as gr
import webbrowser


net_g = None


def get_text(text, language_str, hps):
    norm_text, phone, tone, word2ph = clean_text(text, language_str)
    phone, tone, language = cleaned_text_to_sequence(phone, tone, language_str)

    if hps.data.add_blank:
        phone = commons.intersperse(phone, 0)
        tone = commons.intersperse(tone, 0)
        language = commons.intersperse(language, 0)
        for i in range(len(word2ph)):
            word2ph[i] = word2ph[i] * 2
        word2ph[0] += 1
    bert = get_bert(norm_text, word2ph, language_str)
    del word2ph

    assert bert.shape[-1] == len(phone)

    phone = torch.LongTensor(phone)
    tone = torch.LongTensor(tone)
    language = torch.LongTensor(language)

    return bert, phone, tone, language

def infer(text, sdp_ratio, noise_scale, noise_scale_w, length_scale, sid):
    global net_g
    bert, phones, tones, lang_ids = get_text(text, "ZH", hps)
    with torch.no_grad():
        x_tst=phones.to(device).unsqueeze(0)
        tones=tones.to(device).unsqueeze(0)
        lang_ids=lang_ids.to(device).unsqueeze(0)
        bert = bert.to(device).unsqueeze(0)
        x_tst_lengths = torch.LongTensor([phones.size(0)]).to(device)
        del phones
        speakers = torch.LongTensor([hps.data.spk2id[sid]]).to(device)
        audio = net_g.infer(x_tst, x_tst_lengths, speakers, tones, lang_ids, bert, sdp_ratio=sdp_ratio
                           , noise_scale=noise_scale, noise_scale_w=noise_scale_w, length_scale=length_scale)[0][0,0].data.cpu().float().numpy()
        del x_tst, tones, lang_ids, bert, x_tst_lengths, speakers
        return audio

def tts_fn(text, speaker, sdp_ratio, noise_scale, noise_scale_w, length_scale):
    with torch.no_grad():
        audio = infer(text, sdp_ratio=sdp_ratio, noise_scale=noise_scale, noise_scale_w=noise_scale_w, length_scale=length_scale, sid=speaker)
    return "Success", (hps.data.sampling_rate, audio)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="./logs/Azuma/G_17400.pth", help="path of your model")
    parser.add_argument("--config_dir", default="./configs/config.json", help="path of your config file")
    parser.add_argument("--share", default=False, help="make link public")
    parser.add_argument("-d", "--debug", action="store_true", help="enable DEBUG-LEVEL log")

    args = parser.parse_args()
    if args.debug:
        logger.info("Enable DEBUG-LEVEL log")
        logging.basicConfig(level=logging.DEBUG)
    hps = utils.get_hparams_from_file(args.config_dir)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    '''
    device = (
        "cuda:0"
        if torch.cuda.is_available()
        else (
            "mps"
            if sys.platform == "darwin" and torch.backends.mps.is_available()
            else "cpu"
        )
    )
    '''
    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model).to(device)
    _ = net_g.eval()

    _ = utils.load_checkpoint(args.model_dir, net_g, None, skip_optimizer=True)

    speaker_ids = hps.data.spk2id
    speakers = list(speaker_ids.keys())
    with gr.Blocks() as app:
        with gr.Row():
            with gr.Column():
                gr.Markdown(value="""
                【AI東雪蓮】在线语音合成（Bert-Vits2）\n
                作者：Xz乔希 https://space.bilibili.com/5859321\n
                声音归属：東雪蓮Official https://space.bilibili.com/1437582453\n
                Bert-VITS2项目：https://github.com/Stardust-minus/Bert-VITS2\n
                【AI塔菲】语音合成：https://huggingface.co/spaces/XzJosh/Taffy-Bert-VITS2\n
                使用本模型请严格遵守法律法规！\n
                发布二创作品请标注本项目作者及链接、作品使用Bert-VITS2 AI生成！\n                
                """)
                text = gr.TextArea(label="Text", placeholder="Input Text Here",
                                      value="大家好啊，我是东雪莲，今天来点大家想看的东西啊。")
                speaker = gr.Dropdown(choices=speakers, value=speakers[0], label='Speaker')
                sdp_ratio = gr.Slider(minimum=0.1, maximum=1, value=0.2, step=0.1, label='SDP/DP混合比')
                noise_scale = gr.Slider(minimum=0.1, maximum=1, value=0.5, step=0.1, label='感情调节')
                noise_scale_w = gr.Slider(minimum=0.1, maximum=1, value=0.9, step=0.1, label='音素长度')
                length_scale = gr.Slider(minimum=0.1, maximum=2, value=1, step=0.01, label='生成长度')
                btn = gr.Button("点击生成！", variant="primary")
            with gr.Column():
                text_output = gr.Textbox(label="Message")
                audio_output = gr.Audio(label="Output Audio")

        btn.click(tts_fn,
                inputs=[text, speaker, sdp_ratio, noise_scale, noise_scale_w, length_scale],
                outputs=[text_output, audio_output])
    
#    webbrowser.open("http://127.0.0.1:6006")
#    app.launch(server_port=6006, show_error=True)
        
    app.launch(show_error=True)
