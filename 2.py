import torch, torch.nn as nn, torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
z_dim, batch_size, epochs = 100, 128, 30
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])
loader = DataLoader(
    datasets.MNIST("./data", train=True, download=True, transform=transform),
    batch_size=batch_size, shuffle=True)

G = nn.Sequential(
    nn.Linear(z_dim,256), nn.LeakyReLU(0.2),
    nn.Linear(256,512), nn.LeakyReLU(0.2),
    nn.Linear(512,1024), nn.LeakyReLU(0.2),
    nn.Linear(1024,784), nn.Tanh()
).to(device)

D = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784,512), nn.LeakyReLU(0.2),
    nn.Linear(512,256), nn.LeakyReLU(0.2),
    nn.Linear(256,1), nn.Sigmoid()
).to(device)

loss = nn.BCELoss()
optG = optim.Adam(G.parameters(),0.0002,(0.5,0.999))
optD = optim.Adam(D.parameters(),0.0002,(0.5,0.999))

for epoch in range(1,epochs+1):
    for real,_ in loader:
        real = real.to(device)
        bs = real.size(0)

        real_label = torch.ones(bs,1).to(device)
        fake_label = torch.zeros(bs,1).to(device)
      
        z = torch.randn(bs,z_dim).to(device)
        fake = G(z).view(-1,1,28,28)

        lossD = loss(D(real),real_label) + loss(D(fake.detach()),fake_label)
        optD.zero_grad(); lossD.backward(); optD.step()
      
        z = torch.randn(bs,z_dim).to(device)
        fake = G(z).view(-1,1,28,28)

        lossG = loss(D(fake),real_label)
        optG.zero_grad(); lossG.backward(); optG.step()

    print(f"Epoch {epoch}/{epochs}  Loss_D:{lossD.item():.4f}  Loss_G:{lossG.item():.4f}")

    if epoch % 5 == 0:
        with torch.no_grad():
            z = torch.randn(16,z_dim).to(device)
            samples = G(z).view(-1,1,28,28).cpu()*0.5 + 0.5

        fig,ax = plt.subplots(1,16,figsize=(16,2))
        for i in range(16):
            ax[i].imshow(samples[i][0],cmap="gray")
            ax[i].axis("off")
        plt.title(f"Epoch {epoch}")
        plt.show()
