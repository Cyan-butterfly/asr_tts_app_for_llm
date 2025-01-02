// 获取页面元素
const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const transcript = document.getElementById('transcript');

const speechControls = document.getElementById('speechControls');
const playBtn = document.getElementById('playBtn');
const pauseBtn = document.getElementById('pauseBtn');
const cancelBtn = document.getElementById('cancelBtn');

let mediaRecorder;       // MediaRecorder实例
let audioChunks = [];    // 存储音频数据
let currentUtterance = null;    // 当前的 SpeechSynthesisUtterance 实例

// 添加一个全局变量来存储语音列表
let voices = [];

// 添加音频设备检测函数
function checkAudioOutput() {
    return new Promise((resolve) => {
        // 创建一个测试音频
        const testAudio = new Audio();
        testAudio.volume = 0.01; // 设置非常小的音量

        // 使用一个非常短的无声音频
        testAudio.src = 'data:audio/wav;base64,UklGRjIAAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAAABmYWN0BAAAAAAAAABkYXRhAAAAAA==';

        testAudio.onplay = () => {
            console.log('音频设备正常');
            resolve(true);
        };

        testAudio.onerror = () => {
            console.warn('可能存在音频设备问题，请检查系统音频设置');
            resolve(false);
        };

        // 尝试播放
        const playPromise = testAudio.play();
        if (playPromise) {
            playPromise.catch(() => {
                console.warn('音频播放被阻止，请检查浏览器设置');
                resolve(false);
            });
        }
    });
}

// 修改初始化函数
function initSpeechSynthesis() {
    return new Promise(async (resolve) => {
        console.log('开始初始化语音合成...');

        // 检查音频设备
        const audioDeviceOk = await checkAudioOutput();
        if (!audioDeviceOk) {
            console.warn(`
                提示：请检查以下设置：
                1. Windows 系统声音设置
                2. 默认输出设备是否正确
                3. 应用程序音量是否已打开
                4. 浏览器是否有权限播放声音
            `);
        }

        // 先尝试直接获取语音列表
        voices = window.speechSynthesis.getVoices();
        if (voices.length > 0) {
            console.log('语音列表已存在，无需等待加载');
            console.log('可用语音数量:', voices.length);
            resolve(voices);
            return;
        }

        console.log('等待语音列表加载...');
        // 如果语音列表为空，等待加载
        window.speechSynthesis.onvoiceschanged = () => {
            voices = window.speechSynthesis.getVoices();
            console.log('语音列表加载完成');
            console.log('可用语音数量:', voices.length);
            resolve(voices);
        };
    });
}

// 初始化语音合成，只保留一次调用
document.addEventListener('DOMContentLoaded', () => {
    console.log('页面加载完成，开始初始化语音合成');
    initSpeechSynthesis()
        .then(() => {
            console.log('语音合成初始化完成');
            // 可以在这里添加初始化完成后的操作
        })
        .catch(error => {
            console.error('语音合成初始化失败:', error);
        });
});

// 添加播放状态管理
const playbackState = {
    IDLE: 'idle',
    PLAYING: 'playing',
    STOPPING: 'stopping',
    PAUSED: 'paused'
};

let currentState = playbackState.IDLE;
let currentText = '';

// 清理播放状态
async function cleanupPlayback() {
    if (currentState === playbackState.STOPPING) {
        return; // 已经在停止过程中
    }

    currentState = playbackState.STOPPING;
    console.log('正在清理播放状态，当前状态设为 STOPPING');

    try {
        window.speechSynthesis.cancel();
        currentUtterance = null;
        await new Promise(resolve => setTimeout(resolve, 100)); // 等待状态完全重置
        console.log('播放状态已取消');
    } finally {
        currentState = playbackState.IDLE;
        console.log('播放状态已重置为 IDLE');
    }
}

async function speakText(text) {
    // 如果不是空闲状态，不允许开始新的播放
    if (currentState !== playbackState.IDLE) {
        console.log('当前状态不允许播放:', currentState);
        return;
    }

    return new Promise((resolve, reject) => {
        if (!text || typeof text !== 'string') {
            reject(new Error('无效的文本'));
            return;
        }

        // 创建新的语音实例
        const utterance = new SpeechSynthesisUtterance(text);
        currentUtterance = utterance;

        // 动态设置语言和选择合适的语音
        const isChinese = /[\u4e00-\u9fa5]/.test(text);
        if (isChinese) {
            // 查找中文语音
            const huihuiVoice = voices.find(v => v.name === 'Microsoft Huihui - Chinese (Simplified, PRC)');
            if (huihuiVoice) {
                utterance.voice = huihuiVoice;
                console.log('选择的中文语音:', huihuiVoice.name);
            } else {
                console.warn('未找到指定的中文语音，使用默认语音');
            }
            utterance.lang = 'zh-CN';
        } else {
            // 使用默认英语语音
            const defaultVoice = voices.find(v => v.lang.startsWith('en'));
            if (defaultVoice) {
                utterance.voice = defaultVoice;
                console.log('选择的英语语音:', defaultVoice.name);
            } else {
                console.warn('未找到英语语音，使用默认语音');
            }
            utterance.lang = 'en-US';
        }

        // 设置语音参数
        utterance.rate = 1;
        utterance.pitch = 1;
        utterance.volume = 1;

        utterance.onstart = () => {
            currentState = playbackState.PLAYING;
            speechControls.style.display = 'block';
            console.log('语音合成已启动，状态设为 PLAYING');
        };

        utterance.onend = () => {
            currentState = playbackState.IDLE;
            currentUtterance = null;
            console.log('语音合成已结束，状态设为 IDLE');
            resolve();
        };

        utterance.onerror = (event) => {
            currentState = playbackState.IDLE;
            currentUtterance = null;
            console.error('语音合成错误:', event.error);
            reject(event.error);
        };

        // 开始播放
        try {
            window.speechSynthesis.speak(utterance);
            console.log('开始播放文本:', text);
        } catch (error) {
            currentState = playbackState.IDLE;
            currentUtterance = null;
            console.error('语音合成启动失败:', error);
            reject(error);
        }
    });
}


// 添加播放按钮的事件处理函数
const handlePlay = async () => {
    console.log('播放按钮被点击');

    // 如果不是空闲状态，先停止当前播放
    if (currentState !== playbackState.IDLE) {
        console.log('当前状态不允许播放，正在停止当前播放:', currentState);
        await cleanupPlayback();
    }

    try {
        // 获取要播放的文本
        let textToSpeak = currentText || transcript.textContent;
        console.log('要播放的文本:', textToSpeak);

        // 检查文本是否有效
        if (!textToSpeak || 
            ['正在处理...', '正在录音...', '等待录音...', '识别结果显示在这里'].includes(textToSpeak)) {
            console.log('没有可播放的文本');
            return;
        }

        // 保存当前文本
        currentText = textToSpeak;
        console.log('准备播放文本:', textToSpeak);

        // 播放文本
        await speakText(textToSpeak);
    } catch (error) {
        console.error('播放失败:', error);
        await cleanupPlayback();
    }
};

// 添加节流函数
function throttle(func, limit) {
    let inThrottle;
    return function(...args) {
        if (!inThrottle) {
            func.apply(this, args);
            inThrottle = true;
            setTimeout(() => inThrottle = false, limit);
        }
    }
}

// 使用节流包装播放函数
const throttledPlay = throttle(handlePlay, 300);

// 绑定播放按钮的点击事件
playBtn.addEventListener('click', () => {
    throttledPlay();
});

// 绑定取消按钮的点击事件
cancelBtn.addEventListener('click', async () => {
    console.log('取消按钮被点击');
    await cleanupPlayback();
    speechControls.style.display = 'block';
    console.log('已取消播放');
});

// 绑定暂停按钮的点击事件
pauseBtn.addEventListener('click', () => {
    if (currentState === playbackState.PLAYING) {
        window.speechSynthesis.pause();
        currentState = playbackState.PAUSED;
        console.log('已暂停');
    } else if (currentState === playbackState.PAUSED) {
        window.speechSynthesis.resume();
        currentState = playbackState.PLAYING;
        console.log('已恢复播放');
    }
});

// 确保在录音停止后正确处理
mediaRecorder.onstop = async () => {
    console.log('录音已停止，开始处理音频数据。');
    const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
    audioChunks = [];

    transcript.textContent = '正在处理...';
    currentText = ''; // 清除之前的文本

    const formData = new FormData();
    formData.append('file', audioBlob, 'recording.webm');

    try {
        const response = await fetch('http://localhost:8000/transcribe/', {
            method: 'POST',
            body: formData
        });

        if (response.ok) {
            const result = await response.json();
            const recognizedText = result.transcription || '未识别到文本';
            transcript.textContent = recognizedText;
            currentText = recognizedText;
            console.log('识别结果:', recognizedText);

            if (recognizedText && recognizedText !== '未识别到文本') {
                await new Promise(resolve => setTimeout(resolve, 500));
                await speakText(recognizedText);
            }
        } else {
            const errorResult = await response.json();
            transcript.textContent = `识别失败: ${errorResult.detail || '请稍后重试。'}`;
            console.error('识别失败:', errorResult);
            currentText = '';
        }
    } catch (error) {
        console.error('请求错误:', error);
        transcript.textContent = '请求出错，请检查网络。';
        currentText = '';
    } finally {
        startBtn.disabled = false;
        stopBtn.disabled = true;
        console.log('录音处理完毕，按钮状态已重置。');
    }
}; 
