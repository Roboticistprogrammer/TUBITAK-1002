# TensorRT Deployment
1. Convert to TensorRT EngineTriton and DeepStream perform best with TensorRT (.engine) files. You can export your model directly using the Ultralytics CLI or Python API.CLI Export: Run yolo export model=best.pt format=engine device=0 half=True to create an FP16-optimized engine.Python Export: Use model.export(format='engine', half=True) to generate the file.

2. Set Up the Triton Model RepositoryTriton requires a specific folder structure to serve your model.Create a directory named model_repository.Inside it, create a folder for your model (e.g., yolov8_model).Inside that, create a version folder named 1 and place your .engine file there.Create a config.pbtxt file in the yolov8_model folder to define inputs (e.g., shape [3, 640, 640]) and outputs.

3. Configure DeepStream as a ClientDeepStream will act as the client that sends video frames to Triton for inference via the nvinferserver plugin.DeepStream Config: Edit your deepstream_app_config.txt to point to a secondary inference configuration file.Inference Config: In your config_infer_primary_triton.txt, specify the Triton server details (URL, model name) and the input/output tensor names defined in your config.pbtxt.

4. Deployment and ExecutionLaunch Triton Server: Use the official Triton Docker container to start the server and load your model repository.Run DeepStream App: Execute deepstream-app -c deepstream_app_config.txt to start the pipeline and visualize the inference.

Setup:
> Terminal 1
 docker run --runtime=nvidia --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 -v ${PWD}/model_repository:/models nvcr.io/nvidia/deepstream:7.1-triton-multiarch tritonserver --model-repository=/models

>Terminal 2
python triton_client.py