import sys
import os
import asyncio

# Setup sys path so we can import 'app'
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.services.voice import qwen_tts

def run_test():
    print("Testing Qwen TTS integration...")
    text = "Hello! This is a test of the Qwen TTS integration in Money Printer Turbo Extended."
    voice_name = "qwen:default:Default Voice-Neutral"
    voice_rate = 1.0
    voice_file = "/tmp/qwen_test.wav"
    
    sub_maker = qwen_tts(
        text=text,
        voice_name=voice_name,
        voice_rate=voice_rate,
        voice_file=voice_file
    )
    
    if sub_maker:
        print(f"Generated {len(sub_maker.subs)} subtitles:")
        for sub, offset in zip(sub_maker.subs, sub_maker.offset):
            start = offset[0] / 10000000
            end = offset[1] / 10000000
            print(f"[{start:.2f}s - {end:.2f}s] {sub}")
        print(f"Output saved to: {voice_file}")
    else:
        print("Test failed. SubMaker return object was None.")

if __name__ == "__main__":
    run_test()
