from PIL import Image
import argparse
import glob
import os
import numpy as np
import tqdm

# depthcrafter
import gc
import torch

from diffusers.training_utils import set_seed

from depthcrafter.depth_crafter_ppl import DepthCrafterPipeline
from depthcrafter.unet import DiffusersUNetSpatioTemporalConditionModelDepthCrafter

# input arguments
parser = argparse.ArgumentParser(description='Image Test')

parser.add_argument('--input-path', default="/home/ashkanganj/workspace/Object-Rendering/data/frames2", type=str)
parser.add_argument('--output-path', default="/home/ashkanganj/workspace/Object-Rendering/data/reference_predictions/depthcrafter", type=str)
args = parser.parse_args()

path_input = args.input_path
path_output = args.output_path

print("\nInput path:", path_input, sep = '\t')
print("Output path:", path_output, '\n', sep = '\t')

# get the list of images
filenames = glob.glob(os.path.join(path_input, "*.jpg"))

framecount = len(filenames)
print("Number of frames:", framecount)
# 256 320 384 448 512 576 640 704
work_width = 512
work_height = 384

print("\nLoading input frames...")

# create an empty array of the defined dimensions
frames_array = np.empty((0, work_height, work_width, 3))

progress_bar_load = tqdm.tqdm(total=framecount)
 
# read all images and add them to the array one by one
for i in range(framecount):
  filename = filenames[i]
  image = Image.open(filename).convert("RGB") # to get rid of alpha channel
  if i==0:
    original_width = image.width
    original_height = image.height
  
  new_image = image.resize((work_width, work_height), 1)
  
  img_array = np.asarray(new_image)
  img_array_ex = np.expand_dims(img_array, axis=0)
  frames_array = np.concatenate([frames_array, img_array_ex], axis=0)

  progress_bar_load.update()

progress_bar_load.close()

#print("Frames array:", frames_array.shape)

print("\nOriginal resolution:", original_width, original_height, sep="\t")
print("Work resolution:", work_width, work_height, sep="\t")

# now it's time for DepthCrafter pipeline

# load weights of other components from the provided checkpoint
pre_train_path = "stabilityai/stable-video-diffusion-img2vid-xt"
unet_path = "tencent/DepthCrafter"
cpu_offload: str = "sequential" # model / sequential
num_inference_steps: int = 5
guidance_scale: float = 1.0
window_size: int = 120
overlap: int = 25
track_time: bool = False

unet = DiffusersUNetSpatioTemporalConditionModelDepthCrafter.from_pretrained(
    unet_path,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16,
)

print("\nCreating DepthCrafter pipeline")
pipe = DepthCrafterPipeline.from_pretrained(
    pre_train_path,
    unet=unet,
    torch_dtype=torch.float16,
    variant="fp16",
)

# for saving memory, we can offload the model to CPU, or even run the model sequentially to save more memory
if cpu_offload is not None:
  if cpu_offload == "sequential":
      # This will slow, but save more memory
      pipe.enable_sequential_cpu_offload()
  elif cpu_offload == "model":
      pipe.enable_model_cpu_offload()
  else:
      raise ValueError(f"Unknown cpu offload option: {cpu_offload}")
else:
  pipe.to("cuda")

# enable attention slicing and xformers memory efficient attention
try:
  pipe.enable_xformers_memory_efficient_attention()
except Exception as e:
  print(e)
  print("Xformers is not enabled")
pipe.enable_attention_slicing()

print("\nInferencing")
with torch.inference_mode():
  res = pipe(
    frames_array,
    height=frames_array.shape[1],
    width=frames_array.shape[2],
    output_type="np",
    guidance_scale=guidance_scale,
    num_inference_steps=num_inference_steps,
    window_size=window_size,
    overlap=overlap,
    track_time=track_time,
  ).frames[0]

# save as pickle
import pickle


# # convert the three-channel output to a single channel depth map
res = res.sum(-1) / res.shape[-1]

# # normalize the depth map to [0, 1] across the whole video
# res = (res - res.min()) / (res.max() - res.min())

# # create the target folder
os.makedirs(path_output, exist_ok=True)

print("\nSaving output frames...")

progress_bar_save = tqdm.tqdm(total=framecount)
import matplotlib.pyplot as plt
for i in range(framecount):
    
    frame_array = res[i]
    depth_data = {
                "depth_pred_s0_b1hw": torch.tensor(frame_array).unsqueeze(0).unsqueeze(0)}
    with open(os.path.join(path_output,str(i) + '.pickle'), 'wb') as f:
        pickle.dump(depth_data, f)
#   frame_array = res[i]
#   frame_array = frame_array * 255
#   frame_array = frame_array.astype(np.uint8) 
#   new_image = Image.fromarray(frame_array)
  
#   new_image = new_image.resize((original_width, original_height), 1)
  
#   new_image.save(os.path.join(path_output, shortname))

#   progress_bar_save.update()

# progress_bar_save.close()

print('\nDone.')