import argparse
import cv2
import glob
import numpy as np
import os.path
import time

import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt

class CondInst(object):

    def __init__(self, filepath):

        # Load TensorRT engine
        TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)
        with open(filepath, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())

        # Create context
        self.trt_context = engine.create_execution_context()

        names = ["input", "output_labels", "output_scores", "output_masks"]
        self.h_buffers = []
        self.d_memories = []
        self.bindings = [None] * len(names)
        for name in names:
            idx = engine.get_binding_index(name)
            volume = trt.volume(self.trt_context.get_binding_shape(idx))
            dtype = trt.nptype(engine.get_binding_dtype(idx))
            print(idx, volume, dtype)

            # Create host memory buffers
            buffer = cuda.pagelocked_empty(volume, dtype)
            self.h_buffers.append(buffer)

            # Allocate device memory
            memory = cuda.mem_alloc(buffer.nbytes)
            self.d_memories.append(memory)

            # Set binding
            self.bindings[idx] = int(memory)

        # Create a stream in which to copy inputs/outputs and run inference
        self.stream = cuda.Stream()
        return

    def infer(self, images):
        t0 = time.time()

        num_batch, num_chanels, height, width = images.shape

        # Copy input data to host memory buffer
        np.copyto(self.h_buffers[0], images.ravel())

        # Transfer input data to the GPU.
        cuda.memcpy_htod_async(self.d_memories[0], self.h_buffers[0], self.stream)

        # Run inference.
        self.trt_context.execute_async_v2(
            bindings=self.bindings,
            stream_handle=self.stream.handle)

        # Transfer predictions back from the GPU.
        for i in range(1, 4):
            cuda.memcpy_dtoh_async(self.h_buffers[i], self.d_memories[i], self.stream)

        # Synchronize the stream
        self.stream.synchronize()

        # Reshape
        labels = self.h_buffers[1].reshape(num_batch, -1)
        probs = self.h_buffers[2].astype(np.float32).reshape(num_batch, -1)
        masks = self.h_buffers[3].astype(np.float32).reshape(num_batch, -1, height//4, width//4)

        t1 = time.time()
        return labels, probs, masks, t1 - t0

parser = argparse.ArgumentParser(description="Parameters for TensorRT demo")
parser.add_argument('--camera', action='store_true', help="set this option to use camera image for test")
parser.add_argument('--test_image_dir', type=str, default='../test_image', help="path to test image dir (default '../test_image')")
parser.add_argument('--test_depth_dir', type=str, default=None, help="path to test image dir (required only for model with input_channels=4)")
parser.add_argument('--test_output_dir', type=str, default='../test_output', help="path to test output dir (default '../test_output')")
parser.add_argument('--load_engine', type=str, default='../models/model.engine', help="path to trained model (default '../models/model.py')")
args = parser.parse_args()

def infer_and_visualize(image):
    # Preprocessing
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, dsize=(640, 480))
    image_normalized = (image.astype(np.float32) - np.array([0.485, 0.456, 0.406]) * 255.) / (np.array([0.229, 0.224, 0.225]) * 255.)
    image_normalized = image_normalized.transpose(2, 0, 1) # HWC -> CHW
    image_normalized = image_normalized[None,:,:,:] # CHW -> NCHW

    # Read depth image and concatenate to image
    if args.test_depth_dir is not None:
        basename = os.path.basename(image_path)
        basename = os.path.splitext(basename)[0] + '.png'
        path = os.path.join(args.test_depth_dir, basename)
        depth = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
        depth = depth.astype(np.float32)

        _, _, h, w = image_normalized.shape
        rgbd = np.zeros(shape=(1, 4, h, w), dtype=np.float32)
        rgbd[0,:3,:,:] = image_normalized
        rgbd[0,3,:,:] = depth / 1000. # mm to m
        image_normalized = rgbd

    # Perform inference
    labels, probs, masks, t = condinst.infer(image_normalized)
    print("Inference time {} s".format(t))

    # Postprocessing
    score_threshold=0.3
    num_objects, = probs[probs > score_threshold].shape
    print("{} obects detected".format(num_objects))

    probs = probs[0,:num_objects]
    labels = labels[0,:num_objects]
    if num_objects > 0:
        masks = masks[0,:num_objects,:,:]
    else:
        masks = np.zeros((1, 480, 640), dtype=np.float32)

    print("labels {}".format(labels))
    print("probabilities {}".format(probs))

    masks = masks.transpose(1, 2, 0)
    masks = cv2.resize(masks, dsize=(640, 480), interpolation=cv2.INTER_LINEAR)
    masks = (masks > 0.5).astype(np.float32)

    # Add channel dimension if removed by cv2.resize()
    if len(masks.shape) == 2:
        masks = masks[...,None]

    # Visualize masks
    mask_visualize = np.zeros((480, 640, 3), dtype=np.float32)
    for i in range(masks.shape[2]):
        mask_visualize[:,:,0] += masks[:,:,i] * (float(i+1)%5/4)
        mask_visualize[:,:,1] += masks[:,:,i] * (float(i+1)%4/3)
        mask_visualize[:,:,2] += masks[:,:,i] * (float(i+1)%3/2)
    mask_visualize = np.clip(mask_visualize, 0, 1)
    mask_visualize = mask_visualize * 255
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image_visualize = image / 4 + mask_visualize * 3 / 4
    image_visualize = image_visualize.astype(np.uint8)
    return image_visualize

if __name__ == '__main__':

    # Load TensorRT engine
    print("loading TensorRT engine")
    condinst = CondInst(args.load_engine)

    if args.camera:

        cap = cv2.VideoCapture(-1, cv2.CAP_V4L)

        while True:
            # Read camera image
            ret, image = cap.read()
            if not ret:
                print("failed to read camera image")
                break

            image_visualize = infer_and_visualize(image)

            cv2.imshow("test", image_visualize)
            key = cv2.waitKey(100)
            if key & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    else:
        # Get list of image paths
        image_paths = glob.glob(os.path.join(args.test_image_dir, '*.jpg'))
        image_paths += glob.glob(os.path.join(args.test_image_dir, '*.jpeg'))
        image_paths += glob.glob(os.path.join(args.test_image_dir, '*.png'))

        for image_path in image_paths:
            # Load test images
            print("Loading {}".format(image_path))
            image = cv2.imread(image_path)

            image_visualize = infer_and_visualize(image)

            # Save results
            save_path = os.path.join(args.test_output_dir, os.path.basename(image_path))
            print("Saving to {}".format(save_path))
            os.makedirs(args.test_output_dir, exist_ok=True)
            cv2.imwrite(save_path, image_visualize)
