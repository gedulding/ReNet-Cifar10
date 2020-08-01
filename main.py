import torch;
from torch.utils.data import DataLoader;
from torchvision import datasets;
from torchvision import transforms;
from ResNet import ResNet18;
from torch import nn , optim;

batchsize = 64;
Transforms = transforms.Compose([transforms.Resize((32,32)),transforms.ToTensor()]);

def main():
    cifar_train = datasets.CIFAR10('cifar' , True , transform = Transforms , download = True);
    cifar_train = DataLoader(cifar_train , batch_size = batchsize , shuffle = True);
    cifar_test = datasets.CIFAR10('cifar', False, transform=Transforms, download=True);
    cifar_test = DataLoader(cifar_test, batch_size=batchsize, shuffle=True);

    #浏览数据
    x , label = iter(cifar_train).next();
    print('x : ' , x.shape , 'label : ' , label.shape);
    device = torch.device('cuda');
    model = ResNet18().to(device);
    crition = nn.CrossEntropyLoss();       #交叉熵，适用于分类
    optimizer = optim.Adam(model.parameters() , lr = 1e-3);

    for epoch in range(11):
        model.train();
        for batchid , (x ,label) in enumerate(cifar_train):
            x , label = x.to(device) , label.to(device);
            labels = model(x);
            loss = crition(labels , label);

            #backprop
            optimizer.zero_grad();       #梯度清零
            loss.backward();
            optimizer.step();

        print(epoch , ' : ' , loss.item());

        model.eval();       #test模式
        with torch.no_grad():
            total_score = 0;
            total_number = 0;
            for x, label in cifar_test:
                x, label = x.to(device), label.to(device);
                labels = model(x);
                pred = labels.argmax(dim = 1)
                total_score += torch.eq(pred , label).float().sum().item();
                total_number += x.size(0);
            acc = total_score / total_number;
            print('test acc : ' , acc);

    torch.save(model , 'ResNet.pth');

if __name__ == '__main__':
    main();
