import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
from ultralytics import YOLO

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load YOLO model and prepare dataset
image_folder = "Test_Data"
model_yolo = YOLO("yolov8x-pose-p6.pt")
data = []

for label in os.listdir(image_folder):
    label_dir = os.path.join(image_folder, label)
    for img_name in os.listdir(label_dir):
        img_path = os.path.join(label_dir, img_name)

        # Extract keypoints using YOLOv8
        results = model_yolo.predict(img_path, boxes=False, verbose=False)
        for r in results:
            keypoints = r.keypoints.xyn.cpu().numpy()[0]
            keypoints = keypoints.reshape((1, keypoints.shape[0] * keypoints.shape[1]))[0].tolist()
            keypoints.append(img_path)  # Add image path
            keypoints.append(label)     # Add label

            data.append(keypoints)

# Prepare DataFrame and Label Encoding
total_features = len(data[0])
df = pd.DataFrame(data, columns=[f"x{i}" for i in range(total_features)])
df = df.rename({"x34": "image_path", "x35": "label"}, axis=1).dropna()
df = df.iloc[:, 2:]  # Drop unnecessary columns

le = LabelEncoder()
df['label'] = le.fit_transform(df['label'])
num_classes = len(le.classes_)
X = df.drop(["label", "image_path"], axis=1).values
y = df['label'].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
X_train, X_test = torch.tensor(X_train, dtype=torch.float32), torch.tensor(X_test, dtype=torch.float32)
y_train, y_test = torch.tensor(y_train, dtype=torch.long), torch.tensor(y_test, dtype=torch.long)
train_tensor = TensorDataset(X_train, y_train)
test_tensor = TensorDataset(X_test, y_test)
BATCH_SIZE = 32
train_dataloader = DataLoader(dataset=train_tensor, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(dataset=test_tensor, batch_size=BATCH_SIZE, shuffle=False)

# Define Model
class YogaClassifier(nn.Module):
    def __init__(self, num_classes, input_length):
        super().__init__()
        self.layer1 = nn.Linear(in_features=input_length, out_features=64)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.layer2 = nn.Linear(in_features=64, out_features=64)
        self.outlayer = nn.Linear(in_features=64, out_features=num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.layer2(x)
        x = self.activation(x)
        x = self.outlayer(x)
        return x

# Initialize model, optimizer, and loss function
model = YogaClassifier(num_classes=num_classes, input_length=X.shape[1]).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

# Metrics Storage
train_accuracies = []
val_accuracies = []
epoch_losses = []

# Training Loop
epochs = 200
for epoch in tqdm(range(epochs)):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    for X_batch, y_batch in train_dataloader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = loss_fn(outputs, y_batch)
        loss.backward()
        optimizer.step()

        # Accuracy for training batch
        running_loss += loss.item()
        predictions = outputs.argmax(dim=1)
        correct_train += (predictions == y_batch).sum().item()
        total_train += y_batch.size(0)

    train_accuracy = correct_train / total_train
    train_accuracies.append(train_accuracy)
    epoch_losses.append(running_loss / len(train_dataloader))

    # Validation phase
    model.eval()
    correct_val = 0
    total_val = 0
    y_true = []
    y_pred = []
    with torch.no_grad():
        for X_batch, y_batch in test_dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            predictions = outputs.argmax(dim=1)
            correct_val += (predictions == y_batch).sum().item()
            total_val += y_batch.size(0)
            y_true.extend(y_batch.cpu().numpy())
            y_pred.extend(predictions.cpu().numpy())

    val_accuracy = correct_val / total_val
    val_accuracies.append(val_accuracy)
    print(f"Epoch {epoch + 1}/{epochs} - Loss: {running_loss:.4f} - Train Acc: {train_accuracy:.4f} - Val Acc: {val_accuracy:.4f}")

# Save accuracy and loss graphs
plt.figure(figsize=(10, 5))
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("static/accuracy_plot.png")
plt.close()

plt.figure(figsize=(10, 5))
plt.plot(epoch_losses, label='Training Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig("static/loss_plot.png")
plt.close()

# Precision, Recall, F1-score per class
precision = precision_score(y_true, y_pred, average=None)
recall = recall_score(y_true, y_pred, average=None)
f1 = f1_score(y_true, y_pred, average=None)

metrics_df = pd.DataFrame({
    'Class': le.inverse_transform(range(num_classes)),
    'Precision': precision,
    'Recall': recall,
    'F1 Score': f1
})
metrics_df.to_csv("static/class_metrics.csv", index=False)

# Confusion Matrix
conf_matrix = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.savefig("static/confusion_matrix.png")
plt.close()

# Save the trained model
torch.save(model.state_dict(), "static/best.pth")
print("Training complete. Model saved as 'best.pth'")
