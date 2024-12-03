import io

import torch.nn as nn
import torch.utils.data
from PIL import Image

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(100, 64 * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(64 * 8),
            nn.ReLU(True),
            # state size. ``(64*8) x 4 x 4``
            nn.ConvTranspose2d(64 * 8, 64 * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 4),
            nn.ReLU(True),
            # state size. ``(64*4) x 8 x 8``
            nn.ConvTranspose2d(64 * 4, 64 * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 2),
            nn.ReLU(True),
            # state size. ``(64*2) x 16 x 16``
            nn.ConvTranspose2d(64 * 2, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # state size. ``(64) x 32 x 32``
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. ``(nc) x 64 x 64``
        )

    def forward(self, input):
        return self.main(input)

    @classmethod
    def load(cls, path, device='cpu'):
        model = cls()
        checkpoint = torch.load(path, map_location=torch.device(device), weights_only=False)
        model.load_state_dict(checkpoint)
        model.eval()
        return model


# generator = Generator.load("./dcgan/generator_4_checkpoint_90.pth")
generator = Generator.load("./dcgan/facedcgan_generator.pth")


def generate_single_image(generator, latent_dim=100):
    generator.eval()
    z = torch.randn(1, latent_dim, 1, 1)

    # Generează imaginea
    with torch.no_grad():
        generated_image = generator(z).squeeze(0)  # Elimină dimensiunea batch-ului

    generated_image = (generated_image.permute(1, 2, 0).cpu().numpy() + 1) * 127.5
    generated_image = generated_image.clip(0, 255).astype("uint8")

    pil_image = Image.fromarray(generated_image)

    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    buffer.seek(0)

    return buffer
