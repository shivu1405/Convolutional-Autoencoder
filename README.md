# Convolutional Autoencoder for Image Denoising

## AIM

To develop a convolutional autoencoder for image denoising application.

## Problem Statement and Dataset
Image denoising is an important task in computer vision where the objective is to remove noise from corrupted images while preserving important features.

In this experiment, a Convolutional Autoencoder is used to learn the mapping between noisy images and their corresponding clean images.

The dataset used is the MNIST handwritten digit dataset, which consists of grayscale images of size 28 × 28 pixels. Artificial noise is added to the images, and the autoencoder learns to reconstruct the original clean images.

## DESIGN STEPS
### STEP 1
Import necessary libraries such as TensorFlow/Keras, NumPy, and Matplotlib. Load the dataset and normalize the pixel values.

### STEP 2
Add noise to the images and build a Convolutional Autoencoder model consisting of an encoder and decoder using convolution and upsampling layers.

### STEP 3
Train the model using the noisy images as input and clean images as output, then visualize the original, noisy, and reconstructed images.



## PROGRAM
### Name: Shivasri
### Register Number: 212224220098

```
class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),  # 28 → 14
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), # 14 → 7
            nn.ReLU()
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1), # 7 → 14
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),  # 14 → 28
            nn.Sigmoid()
        )

    def forward(self, x):

        x = self.encoder(x)

        x = self.decoder(x)

        return x
model = DenoisingAutoencoder().to(device)

criterion = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)

def train(model, loader, criterion, optimizer, epochs=5):

    model.train()

    for epoch in range(epochs):

        running_loss = 0

        for images, _ in loader:

            images = images.to(device)

            noisy_images = add_noise(images).to(device)

            outputs = model(noisy_images)

            loss = criterion(outputs, images)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(loader)

        print(f"Epoch [{epoch+1}/{epochs}] Loss: {epoch_loss:.4f}")
```



## OUTPUT

### Model Summary

Include your model summary


### Original vs Noisy Vs Reconstructed Image

![Autoencoder Output](https://raw.githubusercontent.com/shivu1405/Convolutional-Autoencoder/f5b2584eeddde100db2b856347c028cb5d343694/Screenshot%202026-03-15%20195226.png)
Include a few sample images here.



## RESULT
