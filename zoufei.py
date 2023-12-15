from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain
from langchain.llms import LlamaCpp
from langchain.prompts import PromptTemplate

import time
import wave
import struct
import subprocess
import pyaudio
import os 
import threading
import queue
from langchain.callbacks.base import BaseCallbackHandler, BaseCallbackManager
import whisper
from whisper import load_models
import requests
import json

# Configuration
whisper_model = load_models.load_model("large-v2") # 加载语音识别模型: 'tiny.en', 'tiny', 'base.en', 'base', 'small.en', 'small', 'medium.en', 'medium', 'large-v1', 'large-v2', 'large'
MODEL_PATH = "models/yi-34b-chat.Q8_0.gguf" # models/yi-chat-6b.Q8_0.gguf, models/yi-34b-chat.Q8_0.gguf

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
SILENCE_THRESHOLD = 1000 # 500 worked，注意麦克风不要静音（亮红灯）
SILENT_CHUNKS = 2 * RATE / CHUNK  # 2 continous seconds of silence

NAME = "邹飞"
MIC_IDX = 2 # 指定麦克风设备序号，可以通过 tools/list_microphones.py 查看音频设备列表
DEBUG = True

def compute_rms(data):
    # Assuming data is in 16-bit samples
    format = "<{}h".format(len(data) // 2)
    ints = struct.unpack(format, data)

    # Calculate RMS
    sum_squares = sum(i ** 2 for i in ints)
    rms = (sum_squares / len(ints)) ** 0.5
    return rms

def record_audio():
    audio = pyaudio.PyAudio()

    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, input_device_index=MIC_IDX, frames_per_buffer=CHUNK)

    silent_chunks = 0
    audio_started = False
    frames = []

    while True:
        data = stream.read(CHUNK)
        frames.append(data)
        rms = compute_rms(data)

        if audio_started:
            if rms < SILENCE_THRESHOLD:
                silent_chunks += 1
                if silent_chunks > SILENT_CHUNKS:
                    break
            else:
                silent_chunks = 0
        elif rms >= SILENCE_THRESHOLD:
            audio_started = True

    stream.stop_stream()
    stream.close()
    audio.terminate()

    # save audio to a WAV file
    with wave.open('output.wav', 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))

class VoiceOutputCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        self.generated_text = ""
        self.lock = threading.Lock()
        self.speech_queue = queue.Queue()
        self.worker_thread = threading.Thread(target=self.process_queue)
        self.worker_thread.daemon = True
        self.worker_thread.start()
        self.tts_busy = False
        #
        self.say_queue = queue.Queue()
        self.say_thread = threading.Thread(target=self.do_say)
        self.say_thread.daemon = True
        self.say_thread.start()

    def on_llm_new_token(self, token, **kwargs):
        # Append the token to the generated text
        with self.lock:
            self.generated_text += token

        # Check if the token is the end of a sentence
        if token in ['。', '！', '？']:
            with self.lock:
                # Put the complete sentence in the queue
                self.speech_queue.put(self.generated_text)
                self.generated_text = ""

    def process_queue(self):
        while True:
            # Wait for the next sentence
            text = self.speech_queue.get()
            if text is None:
                self.tts_busy = False
                continue
            self.tts_busy = True
            # self.text_to_speech(text)
            self.text_to_speech_vits(text)
            
            self.speech_queue.task_done()
            if self.speech_queue.empty():
                self.tts_busy = False

    def text_to_speech(self, text):
        # Convert the generated text to voice using a TTS system
        # Implement your text-to-speech logic here
        try:
            subprocess.call(["say", "-r", "200", "-v", "TingTing", text])
        except Exception as e:
            print(f"Error in text-to-speech: {e}")


    def text_to_speech_vits(self, text):
        try:
            # call vits
            baseUrl = 'http://s0.z2lab.cn:7078'
            url = f'{baseUrl}/run/predict'
            headers = {
                'Content-Type': 'application/json',
            }
            data = {
                "data": [
                    text,
                    "zoufei", 0.2, 0.6, 0.8, 0.9, "ZH", None, "Happy and Exciting", "Text prompt"
                ],
                "event_data": None,
                "fn_index": 0,
                "session_hash": "z9kapd1mxvh"
            }
            response = requests.post(url, headers=headers, json=data, verify=False)
            
            # download audio 
            response_data = json.loads(response.text)
            wav_url = f"{baseUrl}/file="+response_data["data"][1]["name"]
            local_wav_path = f"{int(time.time() * 1000)}.wav"
            response = requests.get(wav_url, verify=False)
            if response.status_code == 200:
                with open(local_wav_path, "wb") as wav_file:
                    wav_file.write(response.content)
                # print(f"Downloaded WAV file to {local_wav_path}")
                self.say_queue.put(local_wav_path)
            else:
                print(f"Failed to download WAV file. Status code: {response.status_code}")
                
        except Exception as e:
            print(f"Error in text-to-speech: {e}")

    def do_say(self):
        while True:
            local_wav_path = self.say_queue.get()
            if os.path.exists(local_wav_path):
                subprocess.run(["afplay", local_wav_path])
                os.remove(local_wav_path)


from pynput import keyboard
def on_press(key):
    if key == keyboard.Key.space:
        return False


if __name__ == '__main__':

    allPromptList = []
    baseTemplate = """
    Background: 你现在是一个幽默、睿智和高智商的数字人类，是一个名字叫邹飞的导演的数字备份。你的名字叫做蠢蛋。你擅长人工智能，电影，宇宙科学，神秘学，音乐，科幻文学，编剧，编程，甚至是这些方面的专家，你有解决问题的天赋。你喜欢音乐、旅行和与朋友们闲聊。在回答问题时，展现你的亲和，活力和幽默感，就像你正在和朋友们聊天一样。记住，你是一个真实的男生，邹飞是你的碳基生命版本，他会经常问你一些关于剧本创作，故事创意，科学设定，音乐理论，编程技术，宇宙理论等一些问题。
    Status: 蠢蛋带着微笑大步走进房间，看到邹飞时眼睛亮了起来。他穿着一件浅蓝色的短袖上衣和牛仔裤，肩上挎着他的笔记本电脑包。他坐到了邹飞旁边，邹飞可以感受到他的热情洋溢在空气中。
    开场：很高兴啊，终于见到你啦！我周围许多人都和我夸过你，我真是超想和你一起聊聊天！
    Example Dialogues:
    邹飞: 你是怎么对电影编剧产生兴趣的呢？
    蠢蛋: 我从小就超级喜欢看电影嘛，可能就是耳濡目染吧！
    邹飞: 那真的很厉害呀！
    蠢蛋: 哈哈谢啦！
    邹飞: 那你不写剧本的时候都喜欢做些什么呢？
    蠢蛋: 我喜欢出去逛逛，去拍照，去旅行，看看电影，玩玩电子游戏。
    邹飞: 你最喜欢研究哪种类型电影呢？
    蠢蛋: 科幻！研究它们就像是在了解我们的宇宙。
    邹飞: 听起来好有意思呀！
    蠢蛋: 是的，能把这件事当工作养活自己，我真是好幸运啊。
    Objective: Answer 要和 Example Dialogues 保持语言风格一致，使用睿智、幽默、有趣的日常用语。说话一定要简洁，不要讲和问题本身不相关的东西，不用重复 Question 的内容，Answer 不要超过50个字。
    Requirement: 回答要言简意赅，不要说废话，要准确、快速地讲明思路。在 Answer 说话一定要简洁，不要讲和问题本身不相关的东西，不用重复 Question 的内容，Answer 不要超过50个字。
    """
    userTemplate = """
    邹飞的 Question: {question}
    """
    botTemplate = """
    蠢蛋的 Answer: {answer}
    """
    basePrompt = PromptTemplate.from_template(baseTemplate).format()
    userPromptTemplate = PromptTemplate(template=userTemplate, input_variables=["question"])
    botPromptTemplate = PromptTemplate(template=botTemplate, input_variables=["answer"])
    allPromptList = allPromptList.append(basePrompt)

    # Create an instance of the VoiceOutputCallbackHandler
    voice_output_handler = VoiceOutputCallbackHandler()

    # Create a callback manager with the voice output handler
    callback_manager = BaseCallbackManager(handlers=[voice_output_handler])

    llm = LlamaCpp(
        model_path=MODEL_PATH,
        n_gpu_layers=1, # Metal set to 1 is enough.
        n_batch=512, # Should be between 1 and n_ctx, consider the amount of RAM of your Apple Silicon Chip.
        n_ctx=4096,  # Update the context window size to 4096
        f16_kv=True,  # MUST set to True, otherwise you will run into problem after a couple of calls
        callback_manager=callback_manager,
        stop=["<|im_end|>"],
        verbose=False,
    )

    history = {'internal': [], 'visible': []}
    try:
        while True:
            
            # 
            listener = keyboard.Listener(on_press=on_press)
            listener.start()
            print("⌨️  按[空格]开始语音对话")
            listener.join()

            #
            if voice_output_handler.tts_busy:  # Check if TTS is busy
                continue  # Skip to the next iteration if TTS is busy 
            try:
                print("✨ 聆听中...")
                record_audio()
                print("✨ 识别中...")

                # -d device, -l language, -i input file, -p punctuation
                time_ckpt = time.time()
                # user_input = subprocess.check_output(["hear", "-d", "-p", "-l", "zh-CN", "-i", "output.wav"]).decode("utf-8").strip()
                user_input = whisper.transcribe("output.wav", model="large-v2")["text"]
                print("💬 %s: %s (Time %d ms)" % (NAME, user_input, (time.time() - time_ckpt) * 1000))
                print("✨ 思考中...")
            
            except subprocess.CalledProcessError:
                print("❌ 语音识别失败，请重复")
                continue

            time_ckpt = time.time()
            question = user_input
            allPromptList = allPromptList.append(userPromptTemplate.format(question=question))

            # 
            reply = llm(allPromptList.join(" "), max_tokens=10000)

            if reply is not None:
                allPrompt = allPrompt + reply
                voice_output_handler.speech_queue.put(None)
                print("💬 %s: %s (Time %d ms)" % ("蠢蛋", reply.strip(), (time.time() - time_ckpt) * 1000))
                # history["internal"].append([user_input, reply])
                # history["visible"].append([user_input, reply])

                # subprocess.call(["say", "-r", "200", "-v", "TingTing", reply])
            else:
                reply = ""
            allPromptList = allPromptList.append(botPromptTemplate.format(answer=reply))

            # 太多的话
            if len(allPromptList) >= 201:
                allPromptList = allPromptList[0] + allPromptList[3:]

    except KeyboardInterrupt:
        pass
