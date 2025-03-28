# AI hw1 2025 Spring
R13922A15 陳星佑


## Environment details
Use conda to build a environment based on the yml file (in the code folder)
```bash
conda env create -f environment.yml
```
cuda version: 12.4


## How to run the code

### Task 1

Please go into the ```task1``` folder

- For BLIP on MSCOCO, BLIP on Flickr30k, Phi-4 on MSCOCO, run the following command

  ```bash
  python3 main.py
  ```

  The result will be stored in ```evaluation_results.txt```

- For Phi-4 on Flickr30k, run the following command

  ```bash
  python3 phi-4-on-flickr30k.py
  ```

  The result will be stored in ```evaluation_results_phi-4_flickr30k.txt```

### Task 2

Please go into the ```task2``` folder

Task 2-1

- Run ```task2-1.py``` with the following parameters
  - ```content_images_folder```: The path of content image folder
  - ```output_folder```: Your desired output image folder
  - ```caption_csv_path```: Your desired caption.csv storing path
    - caption.csv: The output caption of Phi-4 (Just to confirm the result)

- Example:
  
  ```bash
  python task2-1.py \
    --content_images_folder ./content_image \
    --output_folder ./task2-1_output_image \
    --caption_csv_path ./captions.csv
  ```

Task 2-2

- Run ```task2-2.py``` with the following parameters
  - ```content_images_folder```: The path of content image folder
  - ```output_folder```: Your desired output image folder
  - ```caption_csv_path```: Your desired caption.csv storing path
    - caption.csv: The output caption of Phi-4 (Just to confirm the result)

- Example:
  
  ```bash
  python task2-2.py \
    --content_images_folder ./content_image \
    --output_folder ./task2-2_output_image \
    --caption_csv_path ./captions.csv
  ```