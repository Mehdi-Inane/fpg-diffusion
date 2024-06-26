import torch
from positional_embeddings import PositionalEmbedding
import argparse
from ddpm import MLP,NoiseScheduler
from datasets import DataSet
import matplotlib.pyplot as plt

class Block(torch.nn.Module):
    def __init__(self, size: int):
        super().__init__()
        self.ff = torch.nn.Linear(size, size)
        self.act = torch.nn.LeakyReLU()

    def forward(self, x: torch.Tensor):
        return self.act(self.ff(x))




class Discriminator(torch.nn.Module):

    def __init__(self,n_layers,hidden_size,device='cuda',sigmoid=False,time_emb="sinusoidal",input_emb="sinusoidal",emb_size=128,input_size=2):
        super(Discriminator, self).__init__()
        self.device = device
        self.layers = []
        
        
        self.time_mlp = PositionalEmbedding(emb_size, time_emb,device=self.device)
        self.input_mlp1 = PositionalEmbedding(emb_size, input_emb, scale=25.0,device=self.device)
        self.input_mlp2 = PositionalEmbedding(emb_size, input_emb, scale=25.0,device=self.device)

        concat_size = len(self.time_mlp.layer) + input_size
        self.input_layer = torch.nn.Linear(concat_size, hidden_size).to(self.device)
        self.layers.append(self.input_layer)
        self.layers.append(torch.nn.LeakyReLU().to(self.device))
        for _ in range(n_layers):
            self.layers.append(torch.nn.Linear(hidden_size, hidden_size).to(self.device))
            self.layers.append(torch.nn.LeakyReLU().to(self.device))
        self.layers.append(torch.nn.Linear(hidden_size, 1).to(self.device))
        if sigmoid:
            self.layers.append(torch.nn.Sigmoid().to(self.device))
        self.model = torch.nn.Sequential(*self.layers)
    

    def forward(self, x, t):
        #x1_emb = self.input_mlp1(x[:, 0])
        #x2_emb = self.input_mlp2(x[:, 1])
        t_emb = self.time_mlp(t)
        x = torch.cat((x, t_emb), dim=-1)
        x = self.model(x)
        return x
    



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--input_size", type=int, default=2)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--sigmoid", type=bool, default=False)
    parser.add_argument("--time_emb", type=str, default="sinusoidal")
    parser.add_argument("--input_emb", type=str, default="sinusoidal")
    parser.add_argument("--emb_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_timesteps", type=int, default=50)
    parser.add_argument("--beta_schedule", type=str, default="linear", choices=["linear", "quadratic"])
    args = parser.parse_args()
    device = args.device

    # Using diffusion model to generate fake data points
    checkpoint = torch.load('exps/base/model.pth')

    denoiser = MLP(
        hidden_size=checkpoint["hidden_size"],
        hidden_layers=checkpoint["hidden_layers"],
        emb_size=checkpoint["emb_size"],
        time_emb=checkpoint["time_emb"],
        input_emb=checkpoint["input_emb"],
        device=device)

    denoiser.load_state_dict(checkpoint['model_state_dict'])

    noise_scheduler = NoiseScheduler(
        num_timesteps=args.num_timesteps,
        beta_schedule=args.beta_schedule,
        device=device)


    # Loading training data
    real_train_dataset = DataSet(remove=False,total_len=10000)
    real_train_dataset.plot('Real Data')
    real_train_loader = torch.utils.data.DataLoader(real_train_dataset,batch_size=args.batch_size,shuffle=True)
    fake_train_dataset = denoiser.sample(10000,noise_scheduler)[-1]
    fake_train_loader = torch.utils.data.DataLoader(fake_train_dataset,batch_size=args.batch_size,shuffle=True)
    n_epochs = args.epochs
    


    # Evaluation data
    real_eval_dataset = DataSet(remove=False,total_len=1000)
    real_eval_loader = torch.utils.data.DataLoader(real_eval_dataset,batch_size=args.batch_size,shuffle=True)
    fake_eval_dataset = denoiser.sample(1000,noise_scheduler)[-1]
    fake_eval_loader = torch.utils.data.DataLoader(fake_eval_dataset,batch_size=args.batch_size,shuffle=True)


    # Configuring model
    model = Discriminator(n_layers=args.n_layers,input_size=args.input_size,
                          hidden_size=args.hidden_size,device=args.device,sigmoid=args.sigmoid,
                          time_emb = args.time_emb,input_emb=args.input_emb,emb_size=args.emb_size)
    model.to(args.device)

    

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    if args.sigmoid:
        criterion = torch.nn.BCELoss()
    else:
        criterion = torch.nn.BCEWithLogitsLoss()

    # Training
    avg_losses = []
    eval_losses = []
    for epoch in range(n_epochs):
        model.train()
        avg_loss = 0
        for i, (real_batch, fake_batch) in enumerate(zip(real_train_loader,fake_train_loader)):
            real_batch = real_batch.to(args.device)
            fake_batch = fake_batch.to(args.device)
            real_labels = torch.ones(real_batch.size(0),1).to(args.device)
            fake_labels = torch.zeros(fake_batch.size(0),1).to(args.device)
            optimizer.zero_grad()
            noise_real = torch.randn(real_batch.shape).to(device)
            noise_fake = torch.randn(fake_batch.shape).to(device)
            timesteps = torch.randint(
                0, noise_scheduler.num_timesteps, (real_batch.shape[0],)
            ).long()

            noisy = noise_scheduler.add_noise(real_batch, noise_real, timesteps)
            real_output = model(noisy, timesteps.to(device))
            timesteps = torch.randint(
                0, noise_scheduler.num_timesteps, (fake_batch.shape[0],)
            ).long()
            noisy = noise_scheduler.add_noise(fake_batch, noise_fake, timesteps)
            fake_output = model(noisy,timesteps.to(device))
            real_loss = criterion(real_output,real_labels)
            fake_loss = criterion(fake_output,fake_labels)
            loss = real_loss + fake_loss
            avg_loss += loss.item()
            loss.backward()
            optimizer.step()
        avg_loss /= len(real_train_loader)
        avg_losses.append(avg_loss)
        # eval 

        model.eval()
        eval_loss = 0
        for i, (real_batch, fake_batch) in enumerate(zip(real_eval_loader,fake_eval_loader)):
            real_batch = real_batch.to(args.device)
            fake_batch = fake_batch.to(args.device)
            real_labels = torch.ones(real_batch.size(0),1).to(args.device)
            fake_labels = torch.zeros(fake_batch.size(0),1).to(args.device)
            noise_real = torch.randn(real_batch.shape).to(device)
            noise_fake = torch.randn(fake_batch.shape).to(device)
            timesteps = torch.randint(
                0, noise_scheduler.num_timesteps, (real_batch.shape[0],)
            ).long()

            noisy = noise_scheduler.add_noise(real_batch, noise_real, timesteps)
            real_output = model(noisy, timesteps.to(device))
            timesteps = torch.randint(
                0, noise_scheduler.num_timesteps, (fake_batch.shape[0],)
            ).long()
            noisy = noise_scheduler.add_noise(fake_batch, noise_fake, timesteps)
            fake_output = model(noisy,timesteps.to(device))
            real_loss = criterion(real_output,real_labels)
            fake_loss = criterion(fake_output,fake_labels)
            loss = real_loss + fake_loss
            eval_loss += loss.item()
        eval_loss /= len(real_eval_loader)
        eval_losses.append(eval_loss)
        print(f'Epoch {epoch}/{n_epochs}, Train Loss {avg_loss}, Eval Loss {eval_loss}')
    plt.figure(figsize=(10,5))
    plt.plot(avg_losses,label='CE Loss')
    plt.plot(eval_losses,label='Eval Loss')
    plt.legend()
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()
    plt.savefig('plots/loss.png')
    print('Done training')
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "hidden_size": args.hidden_size,
            "n_layers": args.n_layers,
            "input_size": args.input_size,
            "sigmoid": args.sigmoid,
            "time_emb": args.time_emb,
            "input_emb": args.input_emb,
            "emb_size": args.emb_size,
        },'discriminator.pth')
    print('Model saved')


    



if __name__ == "__main__":
    main()