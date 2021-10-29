import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import json

def main(start, end, input_dir='/opt/ml/segmentation/input/data/', 
        pred_csv='../../submission/DeepLabV3Plus_resnet50.csv', output_path="output.png"):
    colormap = [[0, 0, 0], [192, 0, 128], [0, 128, 192], [0, 128, 64], [128, 0, 0],
                [64, 0, 128], [64, 0, 192], [192, 128, 64], [192, 192, 128], [64,  64, 128], [128, 0, 192]]
    prediction = pd.read_csv(pred_csv)
    
    num_examples = end-start
    fig, ax = plt.subplots(nrows=num_examples//2+num_examples%2, ncols=4, figsize=(num_examples, 2*num_examples), constrained_layout=True)
    fig.tight_layout()

    for idx, row_num in enumerate(range(start, end)):
        # Original Image PredictionString
        original_img = cv2.resize(cv2.imread(input_dir+prediction.iloc[row_num]['image_id']), dsize= (256,256)).astype(np.uint8)
        ax[idx//2][(idx%2)*2].imshow(original_img)
        ax[idx//2][(idx%2)*2].set_title(f"Orignal Image : {prediction.iloc[row_num]['image_id']}")
        # Groud Truth
        predict_class = list(map(int, prediction['PredictionString'][row_num].split()))
        predict_rgb = np.array(list(map(lambda x: list(colormap[x]), predict_class))).reshape(256,256,3).astype(np.uint8)


        ax[idx//2][(idx%2)*2+1].imshow(cv2.addWeighted(original_img, 0.8, predict_rgb, 1,0))

    custom_lines = [plt.Line2D([0], [0], color=list(map(lambda x: x/255., colormap[i])), lw=4) for i in range(11)]
    fig.legend(custom_lines, ("Backgroud", "General trash", "Paper", "Paper pack",
                            "Metal", "Glass", "Plastic", "Styrofoam",
                            "Plastic bag", "Battery", "Clothing")
            )

    plt.savefig(output_path)
    plt.show()
    
    
if __name__ == '__main__':
    with open("__base__.json", 'r') as f:
        cfg = json.load(f)

    main(
        start = cfg["start"],
        end = cfg["end"],
        input_dir = cfg["input_dir"],
        pred_csv = cfg["pred_csv"],
        output_path = cfg["output_path"]
    )