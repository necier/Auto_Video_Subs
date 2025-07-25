# 视频字幕+翻译工具
> 自动识别并翻译视频字幕，并烧录到视频中

这个工具可以帮助用户自动提取视频的音频，使用 Whisper 模型进行字幕识别，并通过 DeepSeek API 将字幕翻译成简体中文，最终将字幕烧录到视频中。

## **功能**
- 提取视频音频
- 使用 Whisper 模型进行字幕识别
- 通过 DeepSeek API 自动翻译字幕
- 烧录翻译后的字幕到视频中

## **部署**

1. 创建并激活虚拟环境：
    ```bash
    python -m venv venv
    venv\Scripts\activate
    ```
2. 确保你本地有ffmpeg.exe和ffprobe.exe并已添加到环境变量
   ```bash
   例如：目录: D:\ffmpeg\bin
    Mode                 LastWriteTime         Length Name
    ----                 -------------         ------ ----
    -a----         2025/3/10     14:05      148234240 ffmpeg.exe
    -a----         2025/3/10     14:05      148079104 ffplay.exe
    -a----         2025/3/10     14:05      148097536 ffprobe.exe
   
   ```
   
4. 安装依赖：
    ```bash
    pip install -r requirements.txt
    ```

## **运行**

1. 配置你的 API 密钥：
    - 在文件下方``'api_key': 'your api key'``中设置你自己的 DeepSeek API 密钥。

2. 运行脚本：
    ```bash
    python release.py
    ```

    首次运行会在代码文件夹内下载model，耗时较长，请确保网络通畅。

    你将看到提示框，让你选择视频文件。

3. 翻译过程完成后，程序会生成音频文件和字幕文件，以及最终带字幕的视频文件。
