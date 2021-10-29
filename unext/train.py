from tqdm import tqdm
import numpy as np
import torch
import os
from utils import label_accuracy_score, add_hist
from util.wandb_function import wandbWrite

class Train():
    def __init__(self, num_epochs, classes, model, train_loader, val_loader, criterion, device):
        self.num_epochs = num_epochs
        self.category = classes
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.device = device
        
    def train(self, optimizer, saved_dir, filename, saveWandb=False):
        print(f'Start training..')
        n_class = 11
        # best_loss = 9999999
        best_mIoU = 0

        for epoch in range(self.num_epochs):
            self.model.train()

            hist = np.zeros((n_class, n_class))
            for step, (images, masks, _) in enumerate(tqdm(self.train_loader, desc=f"Epoch {epoch+1:02}")):
                images = torch.stack(images)       
                masks = torch.stack(masks).long() 

                # gpu 연산을 위해 device 할당
                images, masks = images.to(self.device), masks.to(self.device)

                # device 할당
                self.model = self.model.to(self.device)

                # inference
                outputs = self.model(images)

                # loss 계산 (cross entropy loss)
                loss = self.criterion(outputs, masks)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                outputs = torch.argmax(outputs, dim=1).detach().cpu().numpy()
                masks = masks.detach().cpu().numpy()

                hist = add_hist(hist, masks, outputs, n_class=n_class)
                acc, acc_cls, mIoU, fwavacc, IoU = label_accuracy_score(hist)


            print(f'Epoch [{epoch+1}/{self.num_epochs}], Step [{step+1}/{len(self.train_loader)}], \
                    Loss: {round(loss.item(),4)}, mIoU: {round(mIoU,4)}')

            # validation 주기에 따른 loss 출력 및 best model 저장
            avrg_loss, val_mIoU, IoU_by_class = self.validation(epoch+1)
            if val_mIoU > best_mIoU:
                print(f"Best performance at epoch: {epoch + 1}")
                print(f"Save model in {saved_dir}")
                best_mIoU = val_mIoU
                self.save_model(saved_dir, file_name=f"{filename}_{epoch+1}_{round(val_mIoU,3)}.pth")
            print()
            if saveWandb:
                wandbWrite(
                    epoch=epoch+1,
                    loss=round(loss.item(),4), 
                    mIoU=round(mIoU,4), 
                    val_loss=round(avrg_loss.item(),4), 
                    val_mIoU=round(val_mIoU,4), 
                    IoU_by_class=IoU_by_class
                )
                

    def save_model(self, saved_dir, file_name):
        if not os.path.isdir(saved_dir):                                                           
            os.mkdir(saved_dir)
        check_point = {'net': self.model.state_dict()}
        output_path = os.path.join(saved_dir, file_name)
        torch.save(self.model, output_path)

    def validation(self, epoch):
        print(f'Start validation #{epoch}')
        self.model.eval()

        with torch.no_grad():
            n_class = 11
            total_loss = 0
            cnt = 0

            hist = np.zeros((n_class, n_class))
            for step, (images, masks, _) in enumerate(self.val_loader):

                images = torch.stack(images)       
                masks = torch.stack(masks).long()  

                images, masks = images.to(self.device), masks.to(self.device)            

                # device 할당
                model = self.model.to(self.device)

                outputs = model(images)
                loss = self.criterion(outputs, masks)
                total_loss += loss
                cnt += 1

                outputs = torch.argmax(outputs, dim=1).detach().cpu().numpy()
                masks = masks.detach().cpu().numpy()

                hist = add_hist(hist, masks, outputs, n_class=n_class)

            acc, acc_cls, mIoU, fwavacc, IoU = label_accuracy_score(hist)
            # IoU_by_class = [{classes : round(IoU,4)} for IoU, classes in zip(IoU , self.category)]
            IoU_by_class = dict()
            for IoU, classes in zip(IoU , self.category):
                IoU_by_class[classes] = round(IoU, 4)

            avrg_loss = total_loss / cnt
            print(f'Validation #{epoch}  Average Loss: {round(avrg_loss.item(), 4)}, Accuracy : {round(acc, 4)}, \
                    mIoU: {round(mIoU, 4)}')
            print(f'IoU by class : {IoU_by_class}')

        return avrg_loss, mIoU, IoU_by_class