#!/usr/bin/env python 
import os
import sys
import glob
import regex
import base64
import subprocess
import multiprocessing

import webvtt
import whisper
from transformers import AutoTokenizer, AutoModel


SUPPORTED_LANGUAGE = ['zh', 'en']
WHISPER_MODEL = 'large-v3'
LLM_MODEL = 'THUDM/chatglm3-6b-32k'
LLM_PROMPT = {
    'zh': '请给以下面的内容添加标点符号，修改错别字，并且按语义进行分段： ',
    'en': 'Please add punctuation to the content below, modify the wrong words, and perform segments according to semantics: '
}
LLM_LENGTH = 20000


def run_cmd_with_list(cmd):
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    (out, err) = proc.communicate()
    return out


def download_available_srt(url):
    result = run_cmd_with_list(['yt-dlp', '--list-subs', url]).decode()
    if 'Available subtitles for' not in result:
        return False
    else:
        result = result.rsplit('[info] Available subtitles for ', 1)[-1].strip().split('\n')[2:]
 
    choice = None
    for line in result:
        full_lan = line.split(' ')[0]
        if full_lan.split('-')[0].lower() in SUPPORTED_LANGUAGE:
            choice = full_lan
            break

    if choice:
        tmp_file = base64.b64encode(url.encode(), b'/-').decode() + '.srt'
        result = run_cmd_with_list(['yt-dlp', '--sub-lang', choice, '--write-sub', '--skip-download', url, '-o', tmp_file])
        results = glob.glob(tmp_file+'*')
        if len(results) == 1:
            return results[0]
        else:
            raise Exception('download srt error: ', url)
    else:
        return None


def convert_srt_to_segs(fp):
    segs = []
    for caption in webvtt.read(fp):
        t = caption.text.strip()
        if t:
            segs.append(t)
    return segs


def download_audio(url):
    tmp_file = base64.b64encode(url.encode(), b'/-').decode() + '.tmp'
    download_cmd = ['yt-dlp', '-f', 'wa', '-x', url, '-o', tmp_file]
    result = run_cmd_with_list(download_cmd)
    for i in glob.glob(tmp_file+'*'):
        return i
    raise Exception('download audio error: ', url)


def convert_audio_to_srt_by_another_process(fp, queue):
    """
    use whisper in the same process will not release vram immediately,
    thus put an llm limit to the llm speed.
    """
    model = whisper.load_model(WHISPER_MODEL)
    result = model.transcribe(fp)
    for seg in result['segments']:
        t = seg['text'].strip()
        if t:
            queue.put(t)


def convert_audio_to_srt(fp):
    q = multiprocessing.Queue()
    p = multiprocessing.Process(target=convert_audio_to_srt_by_another_process, args=(fp, q))
    p.start()
    p.join()

    segs = []
    while not q.empty():
        segs.append(q.get())
    return segs


def convert_text_to_article(segs):
    lan = 'en'
    raw = ''.join(segs)
    if (len(regex.findall(r'\p{Han}', ''.join(segs))) > len(raw) / 2):
        lan = 'zh'

    prompt = LLM_PROMPT.get(lan)

    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL, trust_remote_code=True, device_map='auto')
    model = AutoModel.from_pretrained(LLM_MODEL, trust_remote_code=True, device_map='auto').eval()

    ix = 0
    while ix < len(segs):
        seg_raw = ''
        while ix < len(segs) and (len(seg_raw) + len(segs[ix]) < LLM_LENGTH or len(seg_raw) == 0):
            seg_raw += ' ' + segs[ix]
            ix += 1

        current_length = 0
        for response, history in model.stream_chat(tokenizer, prompt+seg_raw, history=[], top_p=1, temperature=0.01):
            print(response[current_length:], end="", flush=True)
            current_length = len(response)
        print('')


def main(url, keep_cache=False):
    srt = download_available_srt(url)
    if srt:
        segs = convert_srt_to_segs(srt)
        if not keep_cache:
            os.remove(srt)
    else:
        fp = download_audio(url)
        segs = convert_audio_to_srt(fp)
        if not keep_cache:
            os.remove(fp)
    
    convert_text_to_article(segs)


if __name__ == '__main__':
    url = sys.argv[-1]
    main(url)
