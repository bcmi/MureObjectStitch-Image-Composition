# CareCom-Image-Composition

Welcome to scan the WeChat QR code to join the discussion group on image composition/compositing, object insertion/addition.
<p align="center">
<img width="150" height="150" alt="Image" src="https://github.com/user-attachments/assets/8572a7c4-0f62-4a5f-bc06-9a1d9fcb14e3" />
</p>

This is the official repository for the following paper:

> **CareCom: Generative Image Composition with Calibrated Reference Features**  [[paper]](https://ojs.aaai.org/index.php/AAAI/article/download/37278/41240) [[arXiv]](https://arxiv.org/pdf/2511.11060)<br>
>
> Jiaxuan Chen, Bo Zhang, Qingdong He, Jinlong Peng, Li Niu<br>
> Accepted by **AAAI 2026**.

CareCom is an extension of [ObjectStitch](https://github.com/bcmi/ObjectStitch-Image-Composition) to support multiple reference images of one foreground object. Generally, more reference images could lead to better results. 

We release the pretrained model for CareCom, which can achieve good results for common or simple objects. However, the pretrained model is weak in keeping the object identity for the objects with rich details. If you have a few images containing the foreground object, we suggest finetuning CareCom using these images, which can greatly promote the detail preservation. We have also extended [ObjectStitch](https://github.com/bcmi/ObjectStitch-Image-Composition) to [OSInsert](https://github.com/bcmi/OSInsert-Image-Composition), which can achieve both high-fidelity (keep the appearance details of reference image) and high-authenticity (adjust foreground pose/viewpoint according to the background) without object-specific finetuning. 

**Note that in the reference images, the foreground object's length and width should fully extend to the edges of the image (see our example), otherwise the performance would be severely affected.**



<p align='center'>  
  <img src='./figs/multiple_foreground_images.jpg.jpg'  width=75% />
  <img src='./figs/network.jpg'  width=70% />
</p>

## Online Demo

Try this [online demo](http://libcom.ustcnewly.com/) for image composition (object insertion) built upon [libcom](https://github.com/bcmi/libcom) toolbox and have fun!

[![]](https://github.com/user-attachments/assets/87416ec5-2461-42cb-9f2d-5030b1e1b5ec)

## Get Started

### 1.  Dependencies

  - torch==1.11.0
  - pytorch_lightning==1.8.1
  - install dependencies:
    ```bash
    cd ObjectStitch-Image-Composition
    pip install -r requirements.txt
    cd src/taming-transformers
    pip install -e .
    ```

### 2.  Download the Pretrained Models

  - Please download the following files to the ``checkpoints`` folder to create the following file tree:
    ```bash
    checkpoints/
    ├── ObjectStitch.pth
    └── openai-clip-vit-large-patch14
        ├── config.json
        ├── merges.txt
        ├── preprocessor_config.json
        ├── pytorch_model.bin
        ├── tokenizer_config.json
        ├── tokenizer.json
        └── vocab.json
    ```
  - **openai-clip-vit-large-patch14 ([Huggingface](https://huggingface.co/BCMIZB/Libcom_pretrained_models/blob/main/openai-clip-vit-large-patch14.zip) | [ModelScope](https://www.modelscope.cn/models/bcmizb/Libcom_pretrained_models/file/view/master/openai-clip-vit-large-patch14.zip))**.

  - **ObjectStitch.pth ([Huggingface](https://huggingface.co/BCMIZB/Libcom_pretrained_models/blob/main/ObjectStitch.pth) | [ModelScope](https://www.modelscope.cn/models/bcmizb/Libcom_pretrained_models/file/view/master/ObjectStitch.pth))**.
  
### 3. Finetune on Examples
- Please prepare your training data following the examples in `finetuning_examples`
- You can use `MureComDataset` in `ldm/data/open_images.py` to load your data. 
- To finetune objectstitch, you can use `main.py`. For example,
  
  ```py
  python main.py \
    --logdir experiments/objectstitch \
    --name='car_multifg' \
    --num_workers 4 \
    --devices 1 \
    --batch_size 1 \
    --num_nodes 1 \
    --base configs/murecom.yaml \
    --package_name='Car' \
    --fg_name='fg1' \
  ```
    or simply run:
    ```
    sh finetune.sh
    ```
- We finetune the model for 150 epochs by default. You can modify the training settings in `configs/murecom.yaml`. After finetuning, the checkpoints are saved in `logdir`. You can change the path of the checkpoints by modifying this argument.
### 4. Inference on Examples
- Please refer to the [examples](./examples/) folder for data preparation:
  - Use the same filename for paired data (e.g., `background` and `bbox`, `foreground` and `foreground_mask`). 
- To perform image composition using the pretrained model or your finetuned model, you can use `scripts/inference.py`. For example,

    ```
    python scripts/inference.py \
    --outdir results \
    --testdir examples/example1 \
    --num_samples 5 \
    --sample_steps 50 \
    --gpu 0
    ```
    or simply run:
    ```
    sh test.sh
    ```
    The images in the ``examples`` folder are collected from [Murecom](https://github.com/bcmi/DreamCom-Image-Composition) dataset. You can place your own images in the ``examples`` folder. If you want to use your own checkpoints, you need to set `ckpt_dir`.
  
### 5. Visualization Results
We showcase several example results generated by the pretrained model and the finetuned model on Murecom dataset. In each example, from left to right, we show the background image with bounding box to insert the foreground object, the reference images of foreground image, and 5 results using different random seeds. The results in odd rows are obtained using the pretrained model, and the results in even rows are obtained using the finetuned model. 

<p align='center'>  
  <img src='./figs/result.jpg'  width=90% />
</p>


We also provide more results of our CareCom on Murecom dataset through [[Baidu Cloud]](https://pan.baidu.com/s/1nfokVKFfCcboHqZjTStWTA?pwd=m7fq) (code: m7fq) or [[Dropbox]](https://www.dropbox.com/scl/fi/j6qjek76s99zraew4qtx0/MureObjectStitch_FOSCom_results.rar?rlkey=c5ooi8tj0ktoojmn3xadq5ww3&st=ypltz4ch&dl=0). In each image in the folder, from top to bottom, we show the results using the model finetuned for 50, 100, 150, 200 epochs. Finetuning 150 epochs can generally achieve satisfactory results. In some cases, finetuning more epochs (e.g., 200 epochs) is helpful for keeping more details, yet at the risk of distorted content and improper illumination. Finetuning 150 epochs takes about 15 minutes on a single A6000 GPU card. 

In the figure below, we show some example results of our CareCom. In each example, from left to right, we show the background image with specified foreground placement, one example reference image of foreground object, and 5 results using different random seeds. The finetuned model can adjust the view/pose of foreground object flexibly while keeping the identity, and its performance is relatively stable when varying the random seed. However, it may fail to preserve the small details of foreground object. 


<p align='center'>  
  <img src='./figs/more_results.jpg'  width=90% />
</p>


## Citation
If you find this work or code is helpful in your research, please cite:
````
@inproceedings{chen2026carecom,
  title={CareCom: Generative Image Composition with Calibrated Reference Features},
  author={Chen, Jiaxuan and Zhang, Bo and He, Qingdong and Peng, Jinlong and Niu, Li},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2026}
}
````

## Other Resources
+ We summarize the papers and codes of generative image composition: [Awesome-Generative-Image-Composition](https://github.com/bcmi/Awesome-Generative-Image-Composition)
+ We summarize the papers and codes of image composition from all aspects: [Awesome-Image-Composition](https://github.com/bcmi/Awesome-Object-Insertion)
+ We summarize all possible evaluation metrics to evaluate the quality of composite images:  [Composite-Image-Evaluation](https://github.com/bcmi/Composite-Image-Evaluation)
+ We write a comprehensive survey on image composition: [the latest version](https://arxiv.org/pdf/2106.14490.pdf)

