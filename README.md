# Whisper Optimized for Apple Neural Engine

Modification of Whisper from OpenAI to optimize for Apple's Neural Engine. By changing the format of the data flowing through the model and re-writing the attention mechanism to work with nn.Conv2d and Einsum instead of nn.Linear we're able improve performance specifically on ANE.

For more information:

https://machinelearning.apple.com/research/neural-engine-transformers


# Requirements

Running the test script requires Whisper, Ml-ane-transformers and ffpmeg. Everything was written and tested using torch 2.0.


```
sudo apt update && sudo apt install ffmpeg
pip install openai-whisper
pip install git+https://github.com/apple/ml-ane-transformers.git
```

# Converting to Coreml

You can modify the convert_to_coreml code to specify if you want a quantized model, by default it's set to False
```
python convert_to_coreml.py
```