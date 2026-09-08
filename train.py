import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from tqdm import tqdm # 建议直接使用 tqdm 避免 auto 在某些环境下不稳定的问题
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_recall_fscore_support

class ClassificationTrainer:
    def __init__(self, model, train_loader, val_loader, 
                 optimizer, scheduler, criterion, device, save_path):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device
        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)
        
        self.best_risk_f1 = 0.0
        self.history = {
            'train_loss': [], 'val_loss': [], 
            'val_acc': [], 'val_macro_f1': [], 'val_weighted_f1': [],
            'risk_prec': [], 'risk_recall': [], 'risk_f1': []
        }
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        
        # 简化后的 tqdm：移除复杂的 bar_format，防止时间计算错误
        pbar = tqdm(
            self.train_loader, 
            desc="Training", 
            leave=False,        # 建议设为 False，每一轮结束后清除，保持终端整洁
            ncols=100,          # 适当增加宽度
            unit="batch"
        )
        
        for batch in pbar:
            x_text = batch['x_text'].to(self.device)
            x_mask = batch['x_mask'].to(self.device)
            y = batch['y'].to(self.device).long() 
            
            logits, _, _ = self.model(x_text, x_mask)
            loss = self.criterion(logits, y)
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            # 使用 set_postfix 传入字典，tqdm 会自动处理格式，比 set_postfix_str 更稳
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        
        return total_loss / len(self.train_loader)
    
    def validate(self, epoch_idx):
        self.model.eval()
        all_preds = []
        all_targets = []
        total_loss = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                x_text = batch['x_text'].to(self.device)
                x_mask = batch['x_mask'].to(self.device)
                y = batch['y'].to(self.device).long()
                
                logits, _, _ = self.model(x_text, x_mask)
                loss = self.criterion(logits, y)
                total_loss += loss.item()
                
                preds = torch.argmax(logits, dim=1)
                all_preds.append(preds.cpu())
                all_targets.append(y.cpu())
        
        all_preds = torch.cat(all_preds).numpy()
        all_targets = torch.cat(all_targets).numpy()
        
        acc = accuracy_score(all_targets, all_preds)
        macro_f1 = f1_score(all_targets, all_preds, average='macro')
        weighted_f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)
        
        # Risk类指标
        risk_prec, risk_recall, risk_f1, _ = precision_recall_fscore_support(
            all_targets, all_preds, 
            labels=[1], 
            average='binary', 
            zero_division=0
        )
        
        # 兼容处理结果
        risk_prec = risk_prec.item() if hasattr(risk_prec, 'item') else risk_prec
        risk_recall = risk_recall.item() if hasattr(risk_recall, 'item') else risk_recall
        risk_f1 = risk_f1.item() if hasattr(risk_f1, 'item') else risk_f1
        
        print(f"\n📊 Epoch {epoch_idx+1} Confusion Matrix:")
        cm = confusion_matrix(all_targets, all_preds)
        if cm.shape == (2, 2):
            print(f"           Pred_Safe  Pred_Risk")
            print(f"Actual_0: {cm[0][0]:^10} {cm[0][1]:^10}")
            print(f"Actual_1: {cm[1][0]:^10} {cm[1][1]:^10}")
        else:
            print(cm)
                
        return acc, macro_f1, weighted_f1, risk_prec, risk_recall, risk_f1, total_loss / len(self.val_loader)
        
    def train(self, num_epochs, patience=3):  # 新增patience参数（早停耐心值）
        print(f"🚀 Start Classification Training on {self.device}...")
        
        # 早停初始化
        best_val_loss = float('inf')  # 监控验证集Loss（核心）
        stop_count = 0                # 连续不提升的epoch数
        best_risk_f1 = -1.0           # Save the first epoch even when F1 is zero.
        self.best_epoch = None
        
        for epoch in range(num_epochs):
            train_loss = self.train_epoch()
            val_acc, val_macro_f1, val_weighted_f1, risk_prec, risk_recall, risk_f1, val_loss = self.validate(epoch)
            
            # Scheduler step（基于验证集Loss）
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # 更新历史记录
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['val_macro_f1'].append(val_macro_f1)
            self.history['val_weighted_f1'].append(val_weighted_f1)
            self.history['risk_prec'].append(risk_prec)
            self.history['risk_recall'].append(risk_recall)
            self.history['risk_f1'].append(risk_f1)
            
            # 打印指标
            print(f"Epoch {epoch+1}/{num_epochs} | "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                f"Acc: {val_acc:.4%} | Risk F1: {risk_f1:.4f} (Prec: {risk_prec:.4f}, Recall: {risk_recall:.4f})")
            
            # 保存最佳模型（基于Risk F1）
            if risk_f1 > best_risk_f1:
                best_risk_f1 = risk_f1
                self.best_epoch = epoch + 1
                torch.save(self.model.state_dict(), self.save_path / "best_model.pt")
                print(f"New Best Risk F1 Saved! (Current Best: {best_risk_f1:.4f})")
            
            # 早停逻辑（监控验证集Loss）
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                stop_count = 0  # 重置计数
                print(f"Validation Loss Improved! (Best: {best_val_loss:.4f})")
            else:
                stop_count += 1
                print(f"Validation Loss Not Improved! Count: {stop_count}/{patience}")
                if stop_count >= patience:
                    print(f"❌ Early Stopping Triggered! Stop at Epoch {epoch+1}")
                    print(f"Best Risk F1: {best_risk_f1:.4f}, Best Val Loss: {best_val_loss:.4f}")
                    break  # 终止训练
        
        print("✅ Training Complete.")
