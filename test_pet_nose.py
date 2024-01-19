import torch, cv2, random, argparse, time
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from pet_nose_loader import PetNoseLoader
from pet_nose_model import PetNoseModel
from torch.utils.data import DataLoader
from scipy.spatial.distance import euclidean


def test_model(model, device, test_loader):
    model.to(device)
    model.eval()
    distances = []
    num_samples = 10
    samples_processed = 0
    total_time = 0.0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            for i, (pred, true) in enumerate(zip(outputs, labels)):
                start_time = time.time()
                dist = euclidean(pred.cpu().numpy(), true.cpu().numpy())
                end_time = time.time()
                total_time += end_time - start_time
                distances.append(dist)

                true_nose = true.cpu().numpy()
                pred_nose = pred.cpu().numpy()

                img = images[i].cpu().numpy().transpose(1, 2, 0)
                img = ((img * 0.5) + 0.5) * 255
                img = img.astype(np.uint8)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) 
                
                true_nose_x, true_nose_y = int(true_nose[0]), int(true_nose[1])
                pred_nose_x, pred_nose_y = int(pred_nose[0]), int(pred_nose[1])
                radius = 5
                cv2.circle(img, (true_nose_x, true_nose_y), radius, (0, 255, 0), 2)
                cv2.circle(img, (pred_nose_x, pred_nose_y), radius, (0, 0, 255), 2)

                if samples_processed < num_samples:
                    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    plt.show()
                    samples_processed += 1

    avg_time_per_image = total_time / len(test_loader.dataset)
    print(f"Average time per image: {avg_time_per_image:.4f} seconds")
    print(f"Minimum Distance: {np.min(distances)}")
    print(f"Mean Distance: {np.mean(distances)}")
    print(f"Maximum Distance: {np.max(distances)}")
    print(f"Standard Distance: {np.std(distances)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', type=int, help="Batch size")
    parser.add_argument('-s', type=str, help="Saved Weight file")
    parser.add_argument('-cuda', type=str, help="Cuda? [Y/N]")
    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() and args.cuda == 'Y' else 'cpu')

    model = PetNoseModel()
    model.load_state_dict(torch.load(args.s))
    model.to(device)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    test_set = PetNoseLoader('../data/images', labels_file='../data/test_noses.txt', transform=transform)
    test_loader = DataLoader(test_set, batch_size=args.b, shuffle=True)

    test_model(model=model, device=device, test_loader=test_loader)

if __name__ == "__main__":
    main()
