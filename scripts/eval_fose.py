import argparse, os, sys, glob
from cgi import test
import cv2
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from imwatermark import WatermarkEncoder
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext
import torchvision
import sys, os
proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, proj_dir)
# os.chdir(proj_dir)
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.data.open_images import FOSEDataset, reverse_image_tensor, reverse_mask_tensor, reverse_clip_tensor
import time
from transformers import AutoFeatureExtractor
import clip
from torchvision.transforms import Resize
from torch.utils.data import DataLoader
from torch.nn import functional as F
import shutil
import copy
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import datetime

def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())

def get_tensor_clip(normalize=True, toTensor=True):
    transform_list = []
    if toTensor:
        transform_list += [torchvision.transforms.ToTensor()]

    if normalize:
        transform_list += [torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                                (0.26862954, 0.26130258, 0.27577711))]
    return torchvision.transforms.Compose(transform_list)

def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"] if "state_dict" in pl_sd else pl_sd
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=True)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)
    model.eval()
    return model


def put_watermark(img, wm_encoder=None):
    if wm_encoder is not None:
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = wm_encoder.encode(img, 'dwtDct')
        img = Image.fromarray(img[:, :, ::-1])
    return img


def load_replacement(x):
    try:
        hwc = x.shape
        y = Image.open("assets/rick.jpeg").convert("RGB").resize((hwc[1], hwc[0]))
        y = (np.array(y)/255.0).astype(x.dtype)
        assert y.shape == x.shape
        return y
    except Exception:
        return x


# def check_safety(x_image):
#     safety_checker_input = safety_feature_extractor(numpy_to_pil(x_image), return_tensors="pt")
#     x_checked_image, has_nsfw_concept = safety_checker(images=x_image, clip_input=safety_checker_input.pixel_values)
#     assert x_checked_image.shape[0] == len(has_nsfw_concept)
#     for i in range(len(has_nsfw_concept)):
#         if has_nsfw_concept[i]:
#             x_checked_image[i] = load_replacement(x_checked_image[i])
#     return x_checked_image, has_nsfw_concept

def get_tensor(normalize=True, toTensor=True):
    transform_list = []
    if toTensor:
        transform_list += [torchvision.transforms.ToTensor()]

    if normalize:
        transform_list += [torchvision.transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
    return torchvision.transforms.Compose(transform_list)

def get_tensor_clip(normalize=True, toTensor=True):
    transform_list = []
    if toTensor:
        transform_list += [torchvision.transforms.ToTensor()]

    if normalize:
        transform_list += [torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                                            (0.26862954, 0.26130258, 0.27577711))]
    return torchvision.transforms.Compose(transform_list)

def prepare_input(batch, model, shape, device, num_samples):
    if num_samples > 1:
        for k in batch.keys():
            if isinstance(batch[k], torch.Tensor):
                batch[k] = torch.cat([batch[k]] * num_samples, dim=0)
    test_model_kwargs={}
    bg_img = batch['inpaint_img'].to(device)
    bg_latent = model.encode_first_stage(bg_img)
    bg_latent = model.get_first_stage_encoding(bg_latent).detach()
    test_model_kwargs['bg_latent'] = bg_latent
    rs_mask = F.interpolate(batch['bg_mask'].to(device), shape[-2:])
    rs_mask = torch.where(rs_mask > 0.5, 1.0, 0.0)
    test_model_kwargs['bbox'] = batch['bbox'].to(device)
    test_model_kwargs['bg_mask']  = rs_mask
    fg_tensor = batch['fg_img'].to(device)
    c = model.get_learned_conditioning(fg_tensor.to(torch.float16))
    c = model.proj_out(c)
    return test_model_kwargs, c

def get_denoise_row_from_list(samples, model, device, desc='', force_no_decoder_quantization=False):
    denoise_row = []
    for zd in tqdm(samples, desc=desc):
        dz = model.decode_first_stage(zd.to(device),force_not_quantize=force_no_decoder_quantization)
        dz = Resize((256, 256))(dz)
        denoise_row.append(dz)
    n_imgs_per_row = len(denoise_row)
    denoise_row = torch.stack(denoise_row)  # n_log_step, n_row, C, H, W
    denoise_grid = rearrange(denoise_row, 'n b c h w -> b n c h w')
    denoise_grid = rearrange(denoise_grid, 'b n c h w -> (b n) c h w')
    denoise_grid = make_grid(denoise_grid, nrow=n_imgs_per_row)
    return denoise_grid

def get_mask_row_from_list(samples, desc=''):
    denoise_row = []
    for dm in tqdm(samples, desc=desc):
        dm = (dm * 2.0 - 1.0)
        dm = Resize((256, 256))(dm)
        denoise_row.append(dm)
    n_imgs_per_row = len(denoise_row)
    denoise_row  = torch.stack(denoise_row)  # n_log_step, n_row, C, H, W
    denoise_grid = rearrange(denoise_row, 'n b c h w -> b n c h w')
    denoise_grid = rearrange(denoise_grid, 'b n c h w -> (b n) c h w')
    denoise_grid = make_grid(denoise_grid, nrow=n_imgs_per_row)
    return denoise_grid

def clip2sd(x):
    # clip input tensor to  stable diffusion tensor
     MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).reshape(1,-1,1,1).to(x.device)
     STD  = torch.tensor([0.26862954, 0.26130258, 0.27577711]).reshape(1,-1,1,1).to(x.device)
     denorm = x * STD + MEAN
     sd_x = denorm * 2 - 1
     return sd_x

def log_dict2tensor(images, max_images=4, clamp=True, target_size=(256, 256)):
    for k in images:
        N = min(images[k].shape[0], max_images)
        images[k] = images[k][:N]
        if isinstance(images[k], torch.Tensor):
            images[k] = images[k].detach().cpu().float()
            if clamp:
                images[k] = torch.clamp(images[k], -1., 1.)
    return images

def draw_bbox_on_background(image_nps, norm_bbox, color=(255,215,0), thickness=3):
    dst_list = []
    for i in range(image_nps.shape[0]):
        img = image_nps[i].copy()
        h,w,_ = img.shape
        x1 = int(norm_bbox[0,0] * w)
        y1 = int(norm_bbox[0,1] * h)
        x2 = int(norm_bbox[0,2] * w)
        y2 = int(norm_bbox[0,3] * h)
        dst = cv2.rectangle(img, (x1, y1), (x2, y2), color=color, thickness=thickness)
        dst_list.append(dst)
    dst_nps = np.stack(dst_list, axis=0)
    return dst_nps


def tensor2numpy(image, normalized=False, image_size=(512, 512)):
    image = Resize(image_size)(image)
    if not normalized:
        image = (image + 1.0) / 2.0  # -1,1 -> 0,1; b,c,h,w
    image = torch.clamp(image, 0., 1.)
    if image.dim() == 3:
        image = image.unsqueeze(0)
    image = image.permute(0, 2, 3, 1)
    image = image.numpy()
    image = (image * 255).astype(np.uint8)
    return image

def save_image(img, img_path):
    img  = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
    flag = cv2.imwrite(img_path, img)
    if not flag:
        print('warning: ', img_path, img.shape)


exp_root = 'experiments/objectstitch_guided/' 
exp_name = '2023-07-14T17-01-11'
ckpt_dir = exp_root + exp_name + '/'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/fose/"
    )
    parser.add_argument(
        "--skip_grid",
        action='store_true',
        help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
    )
    parser.add_argument(
        "--skip_save",
        action='store_true',
        help="do not save individual samples. For speed measurements.",
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )
    parser.add_argument(
        "--fixed_code",
        type=bool,
        # action='store_true',
        default=True,
        help="if enabled, uses the same starting code across samples ",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=2,
        help="sample this often",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--n_imgs",
        type=int,
        default=100,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="how many samples to produce for each given reference image. A.k.a. batch size",
    )
    parser.add_argument(
        "--n_rows",
        type=int,
        default=0,
        help="rows in the grid (default: n_samples)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=5,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=ckpt_dir+"configs/"+exp_name+"-project.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        default=ckpt_dir+"checkpoints/last.ckpt", # epoch=000002.ckpt
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--code_dir",
        type=str,
        default=ckpt_dir+"/code",
        help="path to code",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=23,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )
    opt = parser.parse_args()

    # device = torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu")
    # torch.cuda.set_device(device)

    seed_everything(opt.seed)
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')
    device = torch.device("cuda", local_rank)

    config = OmegaConf.load(f"{opt.config}")
    model  = load_model_from_config(config, f"{opt.ckpt}")
    if local_rank == 0:
        print('complete model building')
    model  = model.to(device)
    if opt.plms:
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)
    
    n_samples = opt.n_samples
    outdir = os.path.join(opt.outdir, f'guidance{opt.scale}')
    # if local_rank == 0 and os.path.exists(outdir):
    #     shutil.rmtree(outdir)
    
    res_dir  = os.path.join(outdir, 'results')
    grid_dir = os.path.join(outdir, 'grid')
    if local_rank == 0:
        os.makedirs(grid_dir, exist_ok=True)
        os.makedirs(res_dir, exist_ok=True)
    dist.barrier()

    n_rows  = opt.n_rows if opt.n_rows > 0 else 1
    dataset = FOSEDataset()
    ddp_sampler = DistributedSampler(dataset)
    dataloader  = DataLoader(dataset, 
                            batch_size=1, 
                            shuffle=False, 
                            num_workers=8,
                            drop_last=False,
                            sampler=ddp_sampler)

    start_code = None
    if opt.fixed_code:
        start_code = torch.randn([n_samples, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)
    precision_scope = autocast if opt.precision=="autocast" else nullcontext
    total_time, total_batch = 0, 0
    if local_rank == 0:
        bar = tqdm(total=len(dataloader))
    else:
        bar = None
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                for index, batch in enumerate(dataloader):
                    start = time.time()
                    if local_rank == 0:
                        bar.update(1)
                    img_name    = os.path.basename(batch['image_path'][0])
                    if os.path.exists(os.path.join(grid_dir, img_name.replace('.png', '.jpg'))):
                        continue
                    shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                    test_model_kwargs, c = prepare_input(copy.deepcopy(batch), model, shape, device, n_samples)
                    uc = None
                    if opt.scale > 1:
                        uc = model.learnable_vector.repeat(c.shape[0], c.shape[1], 1)
                    samples_ddim, z_denoise_row = sampler.sample(S=opt.ddim_steps,
                                                    conditioning=c,
                                                    batch_size=n_samples,
                                                    shape=shape,
                                                    verbose=False,
                                                    eta=opt.ddim_eta,
                                                    x_T=start_code,
                                                    unconditional_guidance_scale=opt.scale,
                                                    unconditional_conditioning=uc,
                                                    test_model_kwargs=test_model_kwargs)
                    # pre_mask = F.interpolate(samples_ddim[:,4:], (opt.H, opt.W))
                    x_samples_ddim = model.decode_first_stage(samples_ddim[:,:4]).cpu().float()
                    res_dict = {} 
                    img_size = (512, 512)

                    res_dict['bg']   = tensor2numpy(batch['bg_img'], image_size=img_size)
                    res_dict['bbox'] = draw_bbox_on_background(res_dict['bg'], batch['bbox'], color=(255,215,0), thickness=3)
                    res_dict['fg']   = tensor2numpy(clip2sd(batch['fg_img']), image_size=img_size)
                    res_dict['comp'] = tensor2numpy(x_samples_ddim, image_size=img_size)
                    respath = os.path.join(res_dir, img_name)
                    save_image(Image.fromarray(res_dict['comp'][0]), respath)

                    x_border = (np.ones((img_size[0], 10, 3)) * 127).astype(np.uint8)
                    grid_img  = [res_dict['bbox'][0], x_border, res_dict['fg'][0], x_border, res_dict['comp'][0]]
                    grid_img = np.concatenate(grid_img, axis=1)
                    grid_img = Image.fromarray(grid_img)
                    scale    = 256. / min(grid_img.size)
                    grid_img = grid_img.resize((int(scale * grid_img.width), int(scale * grid_img.height)))
                    save_image(grid_img, os.path.join(grid_dir, img_name.replace('.png', '.jpg')))

                    if local_rank == 0:
                        per_time = time.time() - start
                        total_time += per_time
                        total_batch += 1
                        avg_time = total_time / total_batch
                        end_time = avg_time * (len(dataloader) - index - 1)
                        last_time = str(datetime.timedelta(seconds=end_time))
                        bar.set_postfix({'avg_time': '{:.1f}s'.format(avg_time)}) # 'last_time': last_time,

    if local_rank == 0:
        print("Your {} samples are ready and waiting for you in {}".format(len(os.listdir(grid_dir)), outdir))


if __name__ == "__main__":
    main()

# torchrun --nproc_per_node=1 --master_port=29501 scripts/eval_cocoee.py