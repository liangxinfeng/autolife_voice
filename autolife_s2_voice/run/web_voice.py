"""浏览器录音 -> 306本地ASR -> LLM任务分析网页。"""
from __future__ import annotations
import argparse, io, json, os, queue, re, ssl, subprocess, sys, threading, time, uuid, wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from autolife_s2_voice.paths import PACKAGE_ROOT, VOID_COG_ROOT

HTTPS_DIR=PACKAGE_ROOT/"configs"/"https"
ASR_SERVICE=None
ASR_LOCK=threading.Lock()
REQUEST_SLOT=threading.BoundedSemaphore(1)
MAX_AUDIO_BYTES=12*1024*1024
MAX_TEXT_CHARS=1000
WEB_READY_MARKER="__WEB_RESULT_READY__"

DISPLAY_LINE = re.compile(r"^\[(?:你|助手|回复|任务\d+|地点\d+)\]\s*.*$")

def display_output(stdout: str, stderr: str, user_text: str = "") -> str:
    """只把用户需要看到的对话和结构化任务行返回给网页。"""
    lines = [line.strip() for line in stdout.splitlines() if DISPLAY_LINE.match(line.strip())]
    lines = [re.sub(r"^\[回复\]", "[助手]", line) for line in lines]
    if user_text and not any(line.startswith("[你]") for line in lines):
        lines.insert(0, f"[你]   {user_text}")
    if lines:
        return "\n".join(lines)
    # 不把模型路径、服务器日志或调用栈回显到平板。
    return (f"[你]   {user_text}\n" if user_text else "") + "[助手] 暂时没有可展示的回复，请重试。"

class RequestError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status=status

def run_agent(text: str) -> tuple[str, int, float, bool]:
    """返回文字结果；marker出现后让TTS留在后台完成并继续占用任务槽。"""
    cmd=[sys.executable,"-m","autolife_s2_voice.run.robot_voice","--text",text,"--no-track","--no-display","--no-ros2","--web-ready-marker"]
    started=time.perf_counter()
    env=os.environ.copy();env["PYTHONUNBUFFERED"]="1"
    process=subprocess.Popen(cmd,cwd=VOID_COG_ROOT,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,bufsize=1,env=env)
    lines: list[str]=[];line_queue: queue.Queue=queue.Queue()

    def read_output():
        try:
            for line in process.stdout or ():
                line_queue.put(line)
        finally:
            line_queue.put(None)

    reader=threading.Thread(target=read_output,name="agent-output",daemon=True);reader.start()
    deadline=time.monotonic()+75;ready=False
    while True:
        remaining=deadline-time.monotonic()
        if remaining<=0:
            process.kill();reader.join(timeout=2)
            raise RequestError(504,"机器人任务分析超时，请稍后重试")
        try:item=line_queue.get(timeout=remaining)
        except queue.Empty:
            process.kill();reader.join(timeout=2)
            raise RequestError(504,"机器人任务分析超时，请稍后重试")
        if item is None:break
        if item.strip()==WEB_READY_MARKER:
            ready=True;break
        lines.append(item)

    elapsed=time.perf_counter()-started
    if not ready:
        reader.join(timeout=2);returncode=process.wait(timeout=3)
        if returncode:print(f"[web] agent error rc={returncode}: {''.join(lines)[-1500:]}",flush=True)
        return display_output("".join(lines),"",text),returncode,elapsed,False

    def finish_in_background():
        reader.join();returncode=process.wait();total=time.perf_counter()-started
        print(f"[web] agent playback finished rc={returncode} total={total:.3f}s",flush=True)
        REQUEST_SLOT.release()

    threading.Thread(target=finish_in_background,name="agent-playback",daemon=True).start()
    return display_output("".join(lines),"",text),0,elapsed,True

def trim_silence(samples):
    """保守裁掉首尾静音，保留180ms边界，降低长空白录音的推理开销。"""
    import numpy as np
    if samples.size<3200:return samples,0.0
    frame=320;count=samples.size//frame
    if not count:return samples,0.0
    framed=samples[:count*frame].reshape(count,frame)
    rms=np.sqrt(np.mean(framed.astype(np.float64)**2,axis=1))
    peak=float(np.max(np.abs(samples)))
    threshold=max(0.0025,min(0.015,peak*0.035))
    active=np.flatnonzero(rms>=threshold)
    if not active.size:return samples,0.0
    margin=2880
    start=max(0,int(active[0])*frame-margin)
    end=min(samples.size,(int(active[-1])+1)*frame+margin)
    # 小于120ms的收益不值得裁剪，也避免频繁切掉轻声首尾。
    removed=(samples.size-(end-start))/16000.0
    return (samples[start:end],max(0.0,removed)) if removed>=0.12 else (samples,0.0)

def recognize_wav(payload: bytes) -> tuple[str, dict]:
    """识别浏览器上传的PCM WAV；ASR模型由网页服务常驻复用。"""
    if ASR_SERVICE is None:raise RuntimeError("本地ASR尚未初始化")
    import numpy as np
    with wave.open(io.BytesIO(payload),"rb") as wav:
        channels=wav.getnchannels();sample_width=wav.getsampwidth();sample_rate=wav.getframerate();frame_count=wav.getnframes()
        if sample_width!=2:raise RequestError(422,"仅支持16位PCM录音")
        if channels<1 or channels>2 or sample_rate<8000 or sample_rate>96000:raise RequestError(422,"录音格式不受支持")
        audio_sec=frame_count/sample_rate
        if audio_sec<0.20:raise RequestError(422,"录音时间太短")
        if audio_sec>65:raise RequestError(413,"单次录音不能超过65秒")
        samples=np.frombuffer(wav.readframes(frame_count),dtype="<i2").astype(np.float32)/32768.0
        if channels>1:samples=samples.reshape(-1,channels).mean(axis=1)
    if not np.isfinite(samples).all():raise RequestError(422,"录音数据无效")
    peak=float(np.max(np.abs(samples))) if samples.size else 0.0
    if peak<0.001:raise RequestError(422,"录音音量过低，请靠近麦克风重试")
    # 网页最终版会直接上传16kHz；兼容旧网页的其他采样率。
    if sample_rate==16000:samples,trimmed_sec=trim_silence(samples)
    else:trimmed_sec=0.0
    started=time.perf_counter()
    with ASR_LOCK:
        result=ASR_SERVICE.recognize(samples,sample_rate=sample_rate)
    asr_sec=time.perf_counter()-started
    text=(result.text or "").strip()
    if not text:raise RequestError(422,"没有识别到有效语音，请靠近麦克风重试")
    return text,{"audio_sec":round(audio_sec,3),"asr_sec":round(asr_sec,3),"input_rate":sample_rate,"trimmed_sec":round(trimmed_sec,3)}

PAGE=r'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>云蝶具身智慧语音中枢</title>
<style>
:root{--c:#5ef5ff;--line:#54e8ff35}*{box-sizing:border-box}body{margin:0;min-height:100vh;color:#e9fbff;font-family:system-ui,"Microsoft YaHei",sans-serif;background:radial-gradient(circle at 50% -10%,#153c68 0,#071326 34%,#020611 75%);overflow-x:hidden}body:before{content:"";position:fixed;inset:0;background-image:linear-gradient(#4eeaff0b 1px,transparent 1px),linear-gradient(90deg,#4eeaff0b 1px,transparent 1px);background-size:38px 38px;pointer-events:none}.orb{position:fixed;border-radius:50%;filter:blur(90px);opacity:.22}.a{width:360px;height:360px;background:#00eaff;left:-140px;top:12%}.b{width:420px;height:420px;background:#703cff;right:-170px;bottom:0}main{width:min(1100px,calc(100% - 28px));margin:auto;padding:34px 0;position:relative}.shell{min-height:calc(100vh - 68px)}header{text-align:center;padding:10px 20px 4px;position:relative}.title{font-size:clamp(28px,4.5vw,48px);font-weight:750;background:linear-gradient(90deg,#fff,#72f5ff 58%,#a890ff);-webkit-background-clip:text;color:transparent}.chip{margin:12px auto 0;font-size:11px;color:#78b7c5;max-width:80%;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.core{padding:22px 30px 14px;text-align:center}.status{font-size:13px;letter-spacing:.12em;color:#8ba8bf;min-height:22px}.status.on{color:var(--c);text-shadow:0 0 18px #5ef5ff80}.mic{position:relative;width:186px;height:186px;margin:18px auto 24px;display:grid;place-items:center}.ring{position:absolute;inset:0;border:1px solid #56f4ff4d;border-radius:50%;animation:spin 10s linear infinite}.ring:before,.ring:after{content:"";position:absolute;border-radius:50%;border:1px dashed #7c72ff4d}.ring:before{inset:13px;animation:spin 7s linear infinite reverse}.ring:after{inset:29px}.mic.listening .ring{box-shadow:0 0 48px #27dcff45}.mic.listening:after{content:"";position:absolute;inset:35px;border-radius:50%;background:#34eaff20;animation:pulse 1.2s ease-out infinite}button{position:relative;z-index:2;width:112px;height:112px;border-radius:50%;border:1px solid #7ff8ff;background:radial-gradient(circle at 35% 30%,#2779aa,#08263f 66%);color:#efffff;font-weight:700;letter-spacing:.1em;cursor:pointer;box-shadow:0 0 28px #24dfff54,inset 0 0 22px #5bf4ff26;transition:.25s}button:hover{transform:scale(1.04);box-shadow:0 0 44px #24dfff80}button.stop{border-color:#ff6b9f;background:radial-gradient(circle at 35% 30%,#9a315b,#351025 66%);box-shadow:0 0 34px #ff3f855c}button:disabled{opacity:.5;cursor:wait}button.fullscreen-btn{position:absolute;right:0;top:0;width:auto;height:38px;padding:0 15px;border-radius:19px;border-color:#62dceb80;background:#071b30cc;font-size:12px;font-weight:650;letter-spacing:.04em;box-shadow:0 0 16px #24dfff25}button.fullscreen-btn:hover{transform:none;background:#0d2b48;box-shadow:0 0 24px #24dfff45}.conversation{text-align:left;width:min(900px,100%);height:270px;margin:0 auto;padding:10px 18px;overflow-y:auto;scrollbar-color:#4fddea55 transparent;scrollbar-width:thin}.result{font:clamp(14px,1.6vw,18px)/1.8 ui-monospace,Consolas,"Microsoft YaHei",monospace;color:#c9deea;white-space:pre-wrap;overflow-wrap:anywhere}.waiting{color:#72dce7;margin-top:8px}.visual{padding:12px 24px 22px}.vh{display:flex;justify-content:space-between;font-size:10px;letter-spacing:.16em;color:#668ca4;margin-bottom:10px}.bars{height:86px;display:flex;align-items:flex-end;justify-content:center;gap:5px}.bar{width:min(12px,2.3vw);height:5%;border-radius:5px 5px 1px 1px;background:linear-gradient(to top,#356cff,#59f5ff 70%,#fff);box-shadow:0 0 10px #4de9ff55;transition:height 70ms linear;opacity:.8}@keyframes spin{to{transform:rotate(360deg)}}@keyframes pulse{from{transform:scale(.65);opacity:.9}to{transform:scale(1.3);opacity:0}}@media(max-width:650px){main{padding:20px 0}.core{padding:20px 8px}.conversation{height:300px;padding:8px}.chip{max-width:95%}button.fullscreen-btn{position:relative;margin:12px auto 0;right:auto;top:auto;height:36px}}
/* 精简记录区：字体略小、可见行数更少，任务和地点突出显示 */
.conversation{width:min(860px,100%);height:210px;padding:8px 18px}.result{font-size:clamp(13px,1.35vw,16px);line-height:1.65;white-space:normal}.result-line{min-height:1.65em;white-space:pre-wrap}.result-line.structured{font-weight:800;color:#e7faff}@media(max-width:650px){.conversation{height:230px}}
</style></head><body><div class="orb a"></div><div class="orb b"></div><main><section class="shell"><header><div class="title">云蝶具身智慧语音中枢</div><div id="device" class="chip">MIC · DETECTING</div><button id="fullscreen" class="fullscreen-btn" type="button" aria-label="进入全屏">⛶ 全屏</button></header><div class="core"><div id="status" class="status">正在初始化本机麦克风</div><div id="mic" class="mic"><div class="ring"></div><button id="control">开始监听</button></div><div id="conversation" class="conversation"><div id="result" class="result"><div class="result-line waiting">请说出您的需求</div></div></div></div><div class="visual"><div class="vh"><span>REAL-TIME AUDIO ENERGY</span><span id="level">LEVEL 000</span></div><div id="bars" class="bars"></div></div></section></main>
<script>
const $=id=>document.getElementById(id);let listening=false,sending=false,stream=null,audioCtx=null,analyser=null,source=null,recorder=null,chunks=[],maxTimer=null,historyText='';const bars=Array.from({length:32},()=>{const b=document.createElement('i');b.className='bar';$('bars').appendChild(b);return b});function status(m,on=false){$('status').textContent=m;$('status').className='status'+(on?' on':'')}function show(content){const result=$('result');while(result.firstChild)result.removeChild(result.firstChild);content.split('\n').forEach(line=>{const row=document.createElement('div');row.className='result-line'+(/^\[(任务|地点)\d+\]/.test(line)?' structured':'');row.textContent=line||' ';result.appendChild(row)});const box=$('conversation');box.scrollTop=box.scrollHeight}
async function prepare(){if(stream)return true;try{stream=await navigator.mediaDevices.getUserMedia({audio:{echoCancellation:true,noiseSuppression:true,autoGainControl:true}});const t=stream.getAudioTracks()[0];$('device').textContent='MIC · '+(t.label||'DEFAULT');audioCtx=new(window.AudioContext||window.webkitAudioContext)();analyser=audioCtx.createAnalyser();analyser.fftSize=128;source=audioCtx.createMediaStreamSource(stream);source.connect(analyser);draw();status('麦克风就绪 · 本地ASR模式');return true}catch(e){status('麦克风不可用 · '+e.message);return false}}
function draw(){if(!analyser)return;const d=new Uint8Array(analyser.frequencyBinCount);analyser.getByteFrequencyData(d);let sum=0;bars.forEach((b,i)=>{const v=d[Math.floor(i*d.length/bars.length)]||0;sum+=v;b.style.height=(listening?Math.max(5,Math.min(100,v*.47+(i%4)*2)):5)+'%'});$('level').textContent='LEVEL '+String(Math.round(sum/bars.length)).padStart(3,'0');requestAnimationFrame(draw)}
function resample(parts,inputRate,targetRate=16000){const count=parts.reduce((n,p)=>n+p.length,0),merged=new Float32Array(count);let at=0;for(const p of parts){merged.set(p,at);at+=p.length}if(inputRate===targetRate)return merged;const ratio=inputRate/targetRate,out=new Float32Array(Math.max(1,Math.floor(count/ratio)));for(let i=0;i<out.length;i++){const start=Math.floor(i*ratio),end=Math.max(start+1,Math.min(count,Math.floor((i+1)*ratio)));let sum=0;for(let j=start;j<end;j++)sum+=merged[j];out[i]=sum/(end-start)}return out}
function encodeWav(parts,inputRate){const sampleRate=16000,samples=resample(parts,inputRate,sampleRate),buffer=new ArrayBuffer(44+samples.length*2),view=new DataView(buffer);let offset=44;const write=(at,s)=>{for(let i=0;i<s.length;i++)view.setUint8(at+i,s.charCodeAt(i))};write(0,'RIFF');view.setUint32(4,36+samples.length*2,true);write(8,'WAVE');write(12,'fmt ');view.setUint32(16,16,true);view.setUint16(20,1,true);view.setUint16(22,1,true);view.setUint32(24,sampleRate,true);view.setUint32(28,sampleRate*2,true);view.setUint16(32,2,true);view.setUint16(34,16,true);write(36,'data');view.setUint32(40,samples.length*2,true);for(let i=0;i<samples.length;i++,offset+=2){const s=Math.max(-1,Math.min(1,samples[i]));view.setInt16(offset,s<0?s*32768:s*32767,true)}return new Blob([buffer],{type:'audio/wav'})}
async function begin(){if(sending||listening)return;if(!await prepare())return;try{await audioCtx.resume();chunks=[];recorder=audioCtx.createScriptProcessor(4096,1,1);recorder.onaudioprocess=e=>{if(listening)chunks.push(new Float32Array(e.inputBuffer.getChannelData(0)))};source.connect(recorder);recorder.connect(audioCtx.destination);listening=true;$('mic').classList.add('listening');$('control').className='stop';$('control').textContent='停止并发送';show(historyText+(historyText?'\n\n':'')+'[你]   正在录音……');status('本地录音中',true);maxTimer=setTimeout(()=>{if(listening)stop()},60000)}catch(e){status('录音启动失败 · '+e.message)}}
function stop(){if(!listening)return;listening=false;clearTimeout(maxTimer);$('mic').classList.remove('listening');$('control').className='';$('control').textContent='开始监听';if(recorder){recorder.disconnect();source.disconnect(recorder);recorder.onaudioprocess=null;recorder=null}const sampleCount=chunks.reduce((n,p)=>n+p.length,0);if(sampleCount<audioCtx.sampleRate*.2){chunks=[];status('录音太短，请重新开始');show(historyText+(historyText?'\n\n':'')+'没有检测到足够长的语音');return}const wav=encodeWav(chunks,audioCtx.sampleRate);chunks=[];submitAudio(wav)}
async function waitPlayback(){for(let i=0;i<120;i++){await new Promise(resolve=>setTimeout(resolve,500));try{const r=await fetch('/api/health',{cache:'no-store'}),d=await r.json();if(!d.busy){$('control').disabled=false;status('机器人播报完成 · 可继续监听');return}}catch(e){}}$('control').disabled=false;status('播报状态未知 · 可以重试')}
async function submitAudio(wav){sending=true;$('control').disabled=true;let playbackPending=false;show(historyText+(historyText?'\n\n':'')+'录音完成，正在进行本地语音识别……');status('306本地ASR识别中',true);const controller=new AbortController(),timer=setTimeout(()=>controller.abort(),90000);try{const r=await fetch('/api/audio',{method:'POST',headers:{'Content-Type':'audio/wav'},body:wav,signal:controller.signal}),d=await r.json(),output=d.output||d.error||'没有返回内容';if(r.ok){historyText+=(historyText?'\n\n':'')+output+'\n已发送，等待下一次语音……'}else{historyText+=(historyText?'\n\n':'')+output}show(historyText);const t=d.timings;playbackPending=Boolean(r.ok&&t&&t.playback_pending);status(r.ok?(playbackPending?`任务结果已返回 · ${t.total_sec.toFixed(2)}秒 · 机器人播报中`:`任务分析完成 · ${t?t.total_sec.toFixed(2)+'秒 · ':''}可继续监听`):'语音处理失败',playbackPending)}catch(e){const msg=e.name==='AbortError'?'处理超时，请稍后重试':'语音上传失败：'+e.message;historyText+=(historyText?'\n\n':'')+msg;show(historyText);status(e.name==='AbortError'?'处理超时':'网络请求失败')}finally{clearTimeout(timer);sending=false;if(playbackPending)waitPlayback();else $('control').disabled=false}}
function activeFullscreen(){return document.fullscreenElement||document.webkitFullscreenElement}
function syncFullscreenButton(){const active=Boolean(activeFullscreen()),button=$('fullscreen');button.textContent=active?'⛶ 退出全屏':'⛶ 全屏';button.setAttribute('aria-label',active?'退出全屏':'进入全屏')}
async function toggleFullscreen(){try{if(activeFullscreen()){const exit=document.exitFullscreen||document.webkitExitFullscreen;if(exit)await exit.call(document)}else{const root=document.documentElement,enter=root.requestFullscreen||root.webkitRequestFullscreen;if(!enter)throw new Error('当前浏览器不支持网页全屏');await enter.call(root)}syncFullscreenButton()}catch(e){status('全屏切换失败 · '+e.message)}}
$('control').onclick=()=>listening?stop():begin();$('fullscreen').onclick=toggleFullscreen;document.addEventListener('fullscreenchange',syncFullscreenButton);document.addEventListener('webkitfullscreenchange',syncFullscreenButton);prepare();
</script></body></html>'''

class VoiceHTTPServer(ThreadingHTTPServer):
    daemon_threads=True
    allow_reuse_address=True
    request_queue_size=8

class Handler(BaseHTTPRequestHandler):
    server_version="AutolifeVoice/2.0"
    def send_body(self,status,body,kind):
        self.send_response(status);self.send_header("Content-Type",kind);self.send_header("Cache-Control","no-store");self.send_header("X-Content-Type-Options","nosniff");self.send_header("Referrer-Policy","no-referrer");self.send_header("Content-Security-Policy","default-src 'self'; style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'; connect-src 'self'; media-src 'self'");self.send_header("Content-Length",str(len(body)));self.end_headers();self.wfile.write(body)
    def send_json(self,status,data):
        self.send_body(status,json.dumps(data,ensure_ascii=False).encode(),"application/json; charset=utf-8")
    def do_GET(self):
        path=self.path.split("?",1)[0]
        if path=="/":self.send_body(200,PAGE.encode(),"text/html; charset=utf-8")
        elif path=="/api/health":self.send_json(200,{"status":"ok","asr_ready":ASR_SERVICE is not None,"busy":REQUEST_SLOT._value==0})
        else:self.send_body(404,b"not found","text/plain; charset=utf-8")
    def do_POST(self):
        request_id=uuid.uuid4().hex[:10];started=time.perf_counter();acquired=False
        try:
            n=int(self.headers.get("Content-Length","0"))
            path=self.path.split("?",1)[0]
            if path not in ("/api/audio","/api/analyze"):
                self.send_body(404,b"not found","text/plain; charset=utf-8");return
            if not REQUEST_SLOT.acquire(blocking=False):raise RequestError(503,"机器人正在处理上一条语音，请稍后重试")
            acquired=True
            if path=="/api/audio":
                if n<44:raise RequestError(422,"录音文件无效")
                if n>MAX_AUDIO_BYTES:raise RequestError(413,"录音文件超过12MB")
                payload=self.rfile.read(n)
                if len(payload)!=n:raise RequestError(400,"录音上传不完整")
                text,timings=recognize_wav(payload);output,returncode,agent_sec,playback_pending=run_agent(text)
                if playback_pending:acquired=False
                timings.update({"agent_ready_sec":round(agent_sec,3),"total_sec":round(time.perf_counter()-started,3),"playback_pending":playback_pending})
                self.send_json(200 if returncode==0 else 502,{"request_id":request_id,"transcript":text,"output":output,"returncode":returncode,"timings":timings})
                print(json.dumps({"event":"voice_request","request_id":request_id,"text":text,"timings":timings,"returncode":returncode},ensure_ascii=False),flush=True);return
            if n<=0 or n>32768:raise RequestError(413,"文本请求过大")
            data=json.loads(self.rfile.read(n));text=str(data.get("text","")).strip()
            if not text or len(text)>MAX_TEXT_CHARS:raise RequestError(422,"文本不能为空且不能超过1000字")
            output,returncode,agent_sec,playback_pending=run_agent(text)
            if playback_pending:acquired=False
            timings={"agent_ready_sec":round(agent_sec,3),"total_sec":round(time.perf_counter()-started,3),"playback_pending":playback_pending}
            self.send_json(200 if returncode==0 else 502,{"request_id":request_id,"output":output,"returncode":returncode,"timings":timings})
            print(json.dumps({"event":"text_request","request_id":request_id,"text":text,"timings":timings,"returncode":returncode},ensure_ascii=False),flush=True)
        except RequestError as e:self.send_json(e.status,{"request_id":request_id,"error":str(e)})
        except (json.JSONDecodeError,UnicodeDecodeError) as e:self.send_json(400,{"request_id":request_id,"error":"请求格式错误"})
        except wave.Error as e:self.send_json(422,{"request_id":request_id,"error":"无法读取WAV录音"})
        except (BrokenPipeError,ConnectionResetError):pass
        except Exception as e:
            print(f"[web] request_id={request_id} unexpected error: {type(e).__name__}: {e}",flush=True)
            self.send_json(500,{"request_id":request_id,"error":"服务器处理失败，请查看306日志"})
        finally:
            if acquired:REQUEST_SLOT.release()
    def log_message(self,fmt,*args):print(f"[web] {self.client_address[0]} {fmt%args}")

def main():
    global ASR_SERVICE
    print("正在加载网页本地ASR模型...")
    from void_cog.services.asr.asr_sherpa import SherpaASRService
    ASR_SERVICE=SherpaASRService();ASR_SERVICE._load_model()
    print("网页本地ASR模型加载完成")
    p=argparse.ArgumentParser();p.add_argument("--host",default="127.0.0.1");p.add_argument("--port",type=int,default=8765);p.add_argument("--https",action="store_true",help="使用 configs/https 下的本地证书启动 HTTPS");a=p.parse_args();s=VoiceHTTPServer((a.host,a.port),Handler);scheme="http"
    if a.https:
        certfile=HTTPS_DIR/"web.crt";keyfile=HTTPS_DIR/"web.key"
        if not certfile.is_file() or not keyfile.is_file():raise FileNotFoundError(f"HTTPS 证书不存在: {HTTPS_DIR}")
        context=ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER);context.load_cert_chain(certfile,keyfile);s.socket=context.wrap_socket(s.socket,server_side=True);scheme="https"
    print(f"网页服务已启动: {scheme}://{a.host}:{a.port}")
    try:s.serve_forever()
    except KeyboardInterrupt:print("正在停止网页服务...")
    finally:s.server_close()
if __name__=="__main__":main()
