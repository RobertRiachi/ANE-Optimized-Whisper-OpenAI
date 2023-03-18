import torch
import coremltools as ct

from coremltools.models.neural_network.quantization_utils import quantize_weights
from whisper import load_model
from model import WhisperANE

def convert_encoder(model, quantize=False):
    model.eval()

    input_shape = (1, 80, 3000)
    input_data = torch.randn(input_shape)
    traced_model = torch.jit.trace(model, input_data)

    model = ct.convert(
        traced_model,
        convert_to=None if quantize else "mlprogram", # convert will fail if weights are quantized, not sure why
        inputs=[ct.TensorType(name="logmel_data", shape=input_shape)],
        compute_units=ct.ComputeUnit.ALL
    )

    if quantize:
        model = quantize_weights(model, nbits=16)

    return model

def convert_decoder(model, quantize=False):
    model.eval()

    tokens_shape = (1, 1)
    audio_shape = (1, 768, 1, 1500)

    audio_data = torch.randn(audio_shape)
    token_data =  torch.randint(50257, tokens_shape).long()
    traced_model = torch.jit.trace(model, (token_data, audio_data))

    model = ct.convert(
        traced_model,
        convert_to=None if quantize else "mlprogram", # convert will fail if weights are quantized, not sure why
        inputs=[
            ct.TensorType(name="token_data", shape=tokens_shape, dtype=int),
            ct.TensorType(name="audio_data", shape=audio_shape)
        ]
    )

    if quantize:
        model = quantize_weights(model, nbits=16)

    return model


if __name__ == "__main__":
    whisper = load_model("small").cpu()
    model_dims = whisper.dims

    whisperANE = WhisperANE(model_dims).eval()
    whisperANE.load_state_dict(whisper.state_dict())

    encoder = whisperANE.encoder
    decoder = whisperANE.decoder

    # Convert ANE encoder
    encoder = convert_encoder(encoder, quantize=False)
    encoder.save(f"encoderANE.mlpackage")

    # Convert ANE decoder
    decoder = convert_decoder(decoder, quantize=False)
    decoder.save("decoderANE.mlpackage")

    print("done converting")
