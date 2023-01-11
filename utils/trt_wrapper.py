import numpy as np
import pycuda
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt


class TRTWarper:
    def __init__(self, engine_file_path: str):
        self.stream = cuda.Stream()
        # self.device = cuda.Device(0)
        # self.ctx = self.device.make_context()

        f = open(engine_file_path, "rb")
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        engine = runtime.deserialize_cuda_engine(f.read())
        self.context = engine.create_execution_context()
        # bindings = []
        assert len(engine) == 3
        # for binding in engine:
        #     binding_idx = engine.get_binding_index(binding)
        #     size = trt.volume(context.get_binding_shape(binding_idx))
        #     dtype = trt.nptype(engine.get_binding_dtype(binding))
        #     if engine.binding_is_input(binding):
        #         input_memory = cuda.mem_alloc(size*np.dtype(dtype).itemsize)
        #         bindings.append(int(input_memory))
        #     else:
        #         output_buffer = cuda.pagelocked_empty(size, dtype)
        #         output_memory = cuda.mem_alloc(output_buffer.nbytes)
        #         bindings.append(int(output_memory))
        # input
        binding = engine[0]
        size = trt.volume(self.context.get_binding_shape(0))
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        self.d_input = cuda.mem_alloc(size * np.dtype(dtype).itemsize)

        # point
        binding = engine[1]
        shape = self.context.get_binding_shape(1)
        size = trt.volume(shape)
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        self.h_point = cuda.pagelocked_empty(size, dtype).reshape(shape[1:])
        self.d_point = cuda.mem_alloc(self.h_point.nbytes)

        # Descriptor
        binding = engine[2]
        shape = self.context.get_binding_shape(2)
        size = trt.volume(shape)
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        self.h_descriptor = cuda.pagelocked_empty(size, dtype).reshape(shape)
        self.d_descriptor = cuda.mem_alloc(self.h_descriptor.nbytes)

        self.bindings = [int(self.d_input), int(self.d_point), int(self.d_descriptor)]

    def __call__(self, img: np.ndarray):
        # self.ctx.push()
        input_buffer = np.ascontiguousarray(img)
        cuda.memcpy_htod_async(self.d_input, input_buffer, self.stream)
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(self.h_point, self.d_point, self.stream)
        cuda.memcpy_dtoh_async(self.h_descriptor, self.d_descriptor, self.stream)
        self.stream.synchronize()
        # self.ctx.pop()
        return self.h_point, self.h_descriptor
