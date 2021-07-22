import cv2
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt
import time

class CenterNetCondInst(object):

    def __init__(self, filepath):

        # Load TensorRT engine
        TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)
        with open(filepath, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())

        # Create context
        self.trt_context = engine.create_execution_context()

        # Determine dimensions and create page-locked memory buffers
        self.h_input = cuda.pagelocked_empty(trt.volume(self.trt_context.get_binding_shape(0)), dtype=np.float32)
        self.h_output0 = cuda.pagelocked_empty(trt.volume(self.trt_context.get_binding_shape(1)), dtype=np.float32)
        self.h_output1 = cuda.pagelocked_empty(trt.volume(self.trt_context.get_binding_shape(2)), dtype=np.int32)
        self.h_output2 = cuda.pagelocked_empty(trt.volume(self.trt_context.get_binding_shape(3)), dtype=np.float32)

        # Allocate device memory for inputs and outputs.
        self.d_input = cuda.mem_alloc(self.h_input.nbytes)
        self.d_output0 = cuda.mem_alloc(self.h_output0.nbytes)
        self.d_output1 = cuda.mem_alloc(self.h_output1.nbytes)
        self.d_output2 = cuda.mem_alloc(self.h_output2.nbytes)

        # Create a stream in which to copy inputs/outputs and run inference
        self.stream = cuda.Stream()
        return

    def infer(self, images):
        t0 = time.time()

        num_batch, num_chanels, height, width = images.shape

        # Copy input data to host memory buffer
        np.copyto(self.h_input, images.ravel())

        # Transfer input data to the GPU.
        cuda.memcpy_htod_async(self.d_input, self.h_input, self.stream)

        # Run inference.
        self.trt_context.execute_async_v2(
            bindings=[int(self.d_input), int(self.d_output0), int(self.d_output1), int(self.d_output2)],
            stream_handle=self.stream.handle)

        # Transfer predictions back from the GPU.
        cuda.memcpy_dtoh_async(self.h_output0, self.d_output0, self.stream)
        cuda.memcpy_dtoh_async(self.h_output1, self.d_output1, self.stream)
        cuda.memcpy_dtoh_async(self.h_output2, self.d_output2, self.stream)

        # Synchronize the stream
        self.stream.synchronize()

        # Reshape
        probs = self.h_output0.reshape(num_batch, -1)
        labels = self.h_output1.reshape(num_batch, -1)
        masks = self.h_output2.reshape(num_batch, -1, height//8, width//8)

        t1 = time.time()
        return probs, labels, masks, t1 - t0

#
# M A I N
if __name__ == '__main__':

    # Load TensorRT engine
    model_filename = "centernet.engine"
    centernet = CenterNetCondInst(model_filename)

    image_filenames = [
        "images/000000041888.jpg",
        "images/000000041990.jpg",
        "images/000000060770.jpg",
    ]

    for filename in image_filenames:

        # Load test images
        print("Loading {}".format(filename))
        image = cv2.imread(filename)

        # Preprocessing
        image = cv2.resize(image, dsize=(640, 480))
        image_normalized = (image.astype(np.float32) / 255. - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        image_normalized = image_normalized.transpose(2, 0, 1) # HWC -> CHW
        image_normalized = image_normalized[None,:,:,:] # CHW -> NCHW

        # Perform inference
        probs, labels, masks, t = centernet.infer(image_normalized)
        print("Inference time {} s".format(t))

        # Postprocessing
        threshold=0.5
        num_objects, = probs[probs > threshold].shape
        print("{} obects detected".format(num_objects))

        probs = probs[0,:num_objects]
        labels = labels[0,:num_objects]
        masks = masks[0,:num_objects,:,:]

        print("labels {}".format(labels))
        print("probabilities {}".format(probs))

        masks = masks.transpose(1, 2, 0)
        masks = cv2.resize(masks, dsize=(640, 480), interpolation=cv2.INTER_NEAREST)

        # Visualize masks
        mask_visualize = np.zeros((480, 640, 3), dtype=np.float32)
        for i in range(masks.shape[2]):
            mask_visualize[:,:,0] += masks[:,:,i] * (float(i+1)%8/7)
            mask_visualize[:,:,1] += masks[:,:,i] * (float(i+1)%4/3)
            mask_visualize[:,:,2] += masks[:,:,i] * (float(i+1)%2/1)
        mask_visualize = np.clip(mask_visualize, 0, 1)
        mask_visualize = (mask_visualize * 255).astype(np.int8)
        image_visualize = image / 2 + mask_visualize / 2
        cv2.imwrite("{}_result.jpg".format(filename), image_visualize)
