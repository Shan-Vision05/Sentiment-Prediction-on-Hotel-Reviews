import torch
import random
import tqdm
import matplotlib.pyplot as plt

### Defining Hyperparameters

loss_function = torch.nn.BCELoss()
num_epochs = 100
batch_size = 16


def GetBatches(X, y, batch_size = 16):
    samples = list(zip(X, y))
    random.shuffle(samples)
    batches = [samples[i:i + batch_size] for i in range(0, len(samples), batch_size)]
    return batches


def TrainStep(model:torch.nn.Module, X_train, y_train):

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    batches = GetBatches(X_train, y_train)

    ## Setting Train environment
    model.train()
    
    train_losses = []
    for batch in tqdm(batches):
        X_Btrain, y_Btrain = zip(*batch)

        # Convert to tensors
        X_Btrain = torch.tensor(X_Btrain, dtype = torch.float32)
        y_Btrain = torch.tensor(y_Btrain, dtype = torch.float32)

        ## Have to increase dimentionality to pass into loss function
        y_Btrain = torch.unsqueeze(y_Btrain, dim = 1) 


        optimizer.zero_grad()

        log_probs = model(X_Btrain)

        loss = loss_function(log_probs, y_Btrain)

        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())
    avg_train_loss = sum(train_losses) / len(train_losses)
    print(f"Average Train Loss: {avg_train_loss}")

    return avg_train_loss


def TestStep(model, X_test, y_test):
    batches = GetBatches(X_test, y_test)

    ## Setting Test environment
    with torch.inference_mode():
        model.eval()

        test_losses = []
        for batch in tqdm(batches):
            X_Btest, y_Btest = zip(*batch)

            # Convert to tensors
            X_Btest = torch.tensor(X_Btest, dtype = torch.float32)
            y_Btest = torch.tensor(y_Btest, dtype = torch.float32)

            ## Have to increase dimentionality to pass into loss function
            y_Btest = torch.unsqueeze(y_Btest, dim = 1) 

            log_probs = model(X_Btest)

            loss = loss_function(log_probs, X_Btest)

            test_losses.append(loss.item())

        avg_test_loss = sum(test_losses) / len(test_losses)
        print(f"Average Test Loss: {avg_test_loss}")
        return avg_test_loss


def Plot_Train_Test_Loss(epochs, train_loss, test_loss):

    epochs = list(range(epochs))  

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, label="Training Loss", marker='o', linestyle='-')
    plt.plot(epochs, test_loss, label="Test Loss", marker='s', linestyle='--')

    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training vs Test Loss")
    plt.legend()
    plt.grid(True)

    plt.show()

def Train(model, X_train, y_train, X_test, y_test, epochs = 100):
    
    train_losses =[]
    test_losses = []
    for epoch in epochs():

        print(f'Epoch: {epoch}')
        train_losses.append(TrainStep(model, X_train, y_train))
        test_losses.append(TestStep(model, X_test, y_test))

    ### Plotting Train and Test Loss Curves
    Plot_Train_Test_Loss(epochs, train_losses, test_losses)

        